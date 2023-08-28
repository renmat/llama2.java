

import java.io.File;
import java.util.List;
import java.util.function.Consumer;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Llama2 {

    private final int dim= 4096;
    private final int hidden_dim= 11008;
    private final int n_layers= 32;
    private final int n_heads= 32;
    private final int n_kv_heads= 32;
    private final int vocab_size= 32000;
    private final int seq_len= 2048;
    private final int head_size = dim/n_heads;
	private final int kv_dim = (dim * n_kv_heads) / n_heads;
	private final int kv_mul = n_heads / n_kv_heads;

    private final float rope_theta = 10000; //1000000 for code-llama
    
    private final INDArray token_embedding_table;
    private final INDArray[] rms_att_weight=new INDArray[n_layers];
    private final INDArray[] rms_ffn_weight=new INDArray[n_layers];
    private final INDArray[] wq=new INDArray[n_layers];
    private final INDArray[] wk=new INDArray[n_layers];
    private final INDArray[] wv=new INDArray[n_layers];
    private final INDArray[] wo=new INDArray[n_layers];
    private final INDArray[] w1=new INDArray[n_layers];
    private final INDArray[] w2=new INDArray[n_layers];
    private final INDArray[] w3=new INDArray[n_layers];
    private final INDArray[] rotary_emb=new INDArray[n_layers];
    private final INDArray rms_final_weight;
    private final INDArray wcls;

    public Llama2(File weightsFolder) {
    	if(!DataType.FLOAT.equals(Nd4j.dataType())) {
    		throw new IllegalStateException();
    	}
    	token_embedding_table = loadWeight(weightsFolder,"model.embed_tokens.weight.npy");    	
    	for(int i=0;i<n_layers;i++) {
    		 rms_att_weight[i]=(loadWeight(weightsFolder,"model.layers."+i+".input_layernorm.weight.npy"));
    		 rms_ffn_weight[i]=(loadWeight(weightsFolder,"model.layers."+i+".post_attention_layernorm.weight.npy"));
    		 wq[i]=(loadWeight(weightsFolder,"model.layers."+i+".self_attn.q_proj.weight.npy")).transpose();
    		 wk[i]=(loadWeight(weightsFolder,"model.layers."+i+".self_attn.k_proj.weight.npy")).transpose();
    		 wv[i]=(loadWeight(weightsFolder,"model.layers."+i+".self_attn.v_proj.weight.npy")).transpose();
    		 wo[i]=(loadWeight(weightsFolder,"model.layers."+i+".self_attn.o_proj.weight.npy")).transpose();
    		 w1[i]=(loadWeight(weightsFolder,"model.layers."+i+".mlp.gate_proj.weight.npy")).transpose();
    		 w2[i]=(loadWeight(weightsFolder,"model.layers."+i+".mlp.down_proj.weight.npy")).transpose();
    		 w3[i]=(loadWeight(weightsFolder,"model.layers."+i+".mlp.up_proj.weight.npy")).transpose();
    		 rotary_emb[i]=(loadWeight(weightsFolder,"model.layers."+i+".self_attn.rotary_emb.inv_freq.npy"));
    	}
    	rms_final_weight = loadWeight(weightsFolder,"model.norm.weight.npy"); 
    	wcls = loadWeight(weightsFolder,"lm_head.weight.npy").transpose();
    	x = Nd4j.zeros(dim);
    	xb = Nd4j.zeros(dim);
    	xb2 = Nd4j.zeros(dim);
    	hb = Nd4j.zeros(hidden_dim);
    	hb2 = Nd4j.zeros(hidden_dim);
    	hbsilu = Nd4j.zeros(hidden_dim);
    	q = Nd4j.zeros(dim);
    	k = Nd4j.zeros(kv_dim);
    	v = Nd4j.zeros(kv_dim);
    	att = Nd4j.zeros(n_heads,seq_len);
    	logits = Nd4j.zeros(vocab_size);
    	key_cache = new INDArray[n_layers][seq_len];
    	value_cache = new INDArray[n_layers][seq_len];
    }
    
    public static INDArray loadWeight(File weightsFolder,String fileName) {
    	INDArray weight = Nd4j.createFromNpyFile(new File(weightsFolder,fileName));
    	weight = weight.castTo(DataType.FLOAT);
    	return weight;
    }
    
	public static double rmsnorm(INDArray output,INDArray input, INDArray weight) {
		if(output==input) {
			throw new IllegalArgumentException();
		}
		output.assign(input);
		output.muli(input);
		double ss = 1.0f / Math.sqrt((output.sum().getFloat() / input.size(0)) + 1e-5f);
		output.assign(input);
		output.muli(ss);
		output.muli(weight);
		return ss;
	}
	
	public static void matmul(INDArray output,INDArray m1,INDArray m2) {
		Nd4j.matmul(m1, m2, output, false, false, false);
	}
    
    private final INDArray x; 
    private final INDArray xb; 
    private final INDArray xb2; 
    private final INDArray hb; 
    private final INDArray hbsilu; 
    private final INDArray hb2; 
    private final INDArray q; 
    private final INDArray k; 
    private final INDArray v; 
    private final INDArray att; 
    private final INDArray logits; 
    
    private final INDArray[][] key_cache;   
    private final INDArray[][] value_cache; 
   
	
	static void softmax(int head, INDArray x, int tokenPosition) {
		INDArray head_scores = x.get(NDArrayIndex.interval(head, head + 1),
				NDArrayIndex.interval(0, tokenPosition + 1));
		Transforms.softmax(head_scores, false);
	}
    
    static void setFloat(INDArray arr,int from, int to, float number) {
    	arr.put(new INDArrayIndex[] {NDArrayIndex.interval(from, to)},number);
    }
    
    public void forward(int token, int tokenPosition) {
    	INDArray tokenEmbedding = token_embedding_table.getRow(token);
    	x.assign(tokenEmbedding);
    	for(int lc=0;lc<n_layers;lc++) {
    		rmsnorm(xb,x,rms_att_weight[lc]);
    		matmul(q, xb, wq[lc]);
    		matmul(k, xb, wk[lc]);
    		matmul(v, xb, wv[lc]);
    		for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0 / Math.pow(rope_theta, head_dim / (float) head_size));
                float val = tokenPosition * freq;
                float fcr = (float) Math.cos(val);
                float fci = (float) Math.sin(val);
                for(INDArray vec:(i < kv_dim?List.of(q,k):List.of(k))) {
                	float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
					vec.putScalar(i, v0 * fcr - v1 * fci);
					vec.putScalar(i + 1, v0 * fci + v1 * fcr);
                }
            }
    		if(key_cache[lc][tokenPosition]==null||value_cache[lc][tokenPosition]==null) {
    			key_cache[lc][tokenPosition] = k.dup();
    			value_cache[lc][tokenPosition]= v.dup();
    		} else {
        		key_cache[lc][tokenPosition].assign(k);
        		value_cache[lc][tokenPosition].assign(v);
    		}
     		INDArray[] layer_k_cache = key_cache[lc];
    		INDArray[] layer_v_cache = value_cache[lc];
			for(int h=0;h<n_heads;h++) {
				int qOffset = h * head_size;
				int cacheOffset = (h / kv_mul) * head_size;
				for (int t = 0; t <= tokenPosition; t++) {
					INDArray layer_token_k_cache = layer_k_cache[t];					
					INDArray q_head = q.get(NDArrayIndex.interval(qOffset, qOffset + head_size));
					INDArray layer_token_k_cache_head = layer_token_k_cache.get(NDArrayIndex.interval(cacheOffset, cacheOffset + head_size));
					float score = (float)(Nd4j.matmul(q_head,layer_token_k_cache_head).getFloat() / Math.sqrt(head_size));
					att.put(h, t, score);
				}
				softmax(h, att, tokenPosition);
				int xbOffset = h * head_size;
				setFloat(xb,xbOffset, xbOffset + head_size, 0f);				
				for (int t = 0; t <= tokenPosition; t++) {
					INDArray layer_token_v_cache = layer_v_cache[t];
                    float a = att.getFloat(h,t);
                    for (int i = 0; i < head_size; i++) {
                       float xbVal = xb.getFloat(xbOffset + i);
                       xbVal+= a * layer_token_v_cache.getFloat(cacheOffset + i);  
                       setFloat(xb,xbOffset+i, xbOffset + i+1,xbVal);
                    }
                }
			}
			
			matmul(xb2, xb, wo[lc]);
			x.addi(xb2);
			rmsnorm(xb, x, rms_ffn_weight[lc]);
			
            matmul(hb, xb, w1[lc]);
            matmul(hb2, xb, w3[lc]);
            
			hbsilu.assign(hb);
			hbsilu.negi();
			Transforms.exp(hbsilu, false);
			hbsilu.addi(1.0f);
			Transforms.pow(hbsilu, -1, false);
			hb.muli(hbsilu);
			

            hb.muli(hb2);
            matmul(xb, hb, w2[lc]);
            x.addi(xb);   
            
            
    	}
    	rmsnorm(xb, x, rms_final_weight);
        matmul(logits, xb, wcls);
    }
    
    public void generate(List<Integer> prompt_tokens,Consumer<Integer> consumer){
        int token = 1;
        int tokenPosition = 0;
        while (tokenPosition < seq_len) {
        	 forward(token, tokenPosition);
        	 if (tokenPosition < prompt_tokens.size()) {
        		 token = prompt_tokens.get(tokenPosition);
             } else {
            	 token = logits.argMax().getInt();//basic for testing
             }
        	 tokenPosition++;
             if (token == 1||token == 2) {
                 break;
             }
             consumer.accept(token);
        }        
    }
}
