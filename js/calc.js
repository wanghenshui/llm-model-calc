document.addEventListener('DOMContentLoaded', function() {
    const jsonInput = document.getElementById('json-input');
    const calculateBtn = document.getElementById('calculate-btn');
    const exampleBtn = document.getElementById('example-btn');
    const totalParamsEl = document.getElementById('total-params');
    const prefillFlopsEl = document.getElementById('prefill-flops');
    const decodeFlopsEl = document.getElementById('decode-flops');
    const ttftEl = document.getElementById('ttft');
    const throughputEl = document.getElementById('throughput');
    const gpuFlopsInput = document.getElementById('gpu-flops');
    const errorMessage = document.getElementById('error-message');
    
    // 示例配置
    const exampleConfig = {
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 27648,
        "max_position_embeddings": 131072,
        "max_window_layers": 64,
        "model_type": "qwen2",
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "sliding_window": 131072,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.1",
        "vocab_size": 152064,
        // 添加计算需要的额外参数
        "prompt_token_length": 1024,
        "output_token_length": 100,
        "batch_size": 1
    };
    
    // 加载示例配置
    exampleBtn.addEventListener('click', function() {
        jsonInput.value = JSON.stringify(exampleConfig, null, 2);
    });
    
    // 计算按钮点击事件
    calculateBtn.addEventListener('click', function() {
        try {
            errorMessage.textContent = '';
            const config = JSON.parse(jsonInput.value);
            const gpuFlops = parseFloat(gpuFlopsInput.value);
            
            if (isNaN(gpuFlops) || gpuFlops <= 0) {
                throw new Error("请输入有效的 GPU FLOPS 值");
            }
            
            // 补全可能缺失的参数
            if (!config.prompt_token_length) config.prompt_token_length = 1024;
            if (!config.output_token_length) config.output_token_length = 100;
            if (!config.batch_size) config.batch_size = 1;
            
            // 计算总参数量
            const totalParams = calculateTotalParameters(config);
            totalParamsEl.textContent = `${totalParams.toFixed(2)} B`;
            
            // 计算prefill阶段FLOPS
            const prefillFlops = calculatePrefillingFLOPs(config);
            prefillFlopsEl.textContent = `${prefillFlops.toFixed(2)} TFLOPs`;
            
            // 计算每个token的解码FLOPS
            const decodeFlops = calculateDecodingFLOPsPerToken(config);
            decodeFlopsEl.textContent = `${decodeFlops.toFixed(2)} TFLOPs`;
            
            // 计算TTFT (Time To First Token)
            const ttft = prefillFlops / gpuFlops;
            ttftEl.textContent = `${ttft.toFixed(3)} 秒`;
            
            // 计算Throughput
            const throughput = gpuFlops / decodeFlops;
            throughputEl.textContent = `${throughput.toFixed(2)} tokens/sec`;
            
        } catch (error) {
            errorMessage.textContent = `计算错误: ${error.message}`;
        }
    });
    
    // 计算总参数量
    function calculateTotalParameters(config) {
        // 计算嵌入层参数量
        const embeddingParams = config.vocab_size * config.hidden_size;
        
        // 计算每层的参数量
        // 前馈网络（FFN）部分
        const ffnParams = 3 * (config.hidden_size * config.intermediate_size);
        
        // 多头注意力机制部分 Q, K, V
        const attentionParams = 2 * config.hidden_size * config.hidden_size * 
                               config.num_key_value_heads/config.num_attention_heads + 
                               config.hidden_size * config.hidden_size;
        
        // 输出投影部分O
        const outputProjectionParams = config.hidden_size * config.hidden_size;
        
        // 每层的总参数量
        const layerParams = ffnParams + attentionParams + outputProjectionParams;
        
        // 总参数量
        const totalParams = embeddingParams + layerParams * config.num_hidden_layers;
        
        return totalParams / 1e9; // 转换为B
    }
    
    // 计算Prefill阶段FLOPS
    function calculatePrefillingFLOPs(config) {
        // Q投影计算量
        const queryProjectionFlops = 2 * config.prompt_token_length * config.hidden_size ** 2;
        
        // K,V投影计算量
        const keyProjectionFlops = 2 * config.prompt_token_length * config.hidden_size ** 2 * 
                                  config.num_key_value_heads/config.num_attention_heads;
        const valueProjectionFlops = 2 * config.prompt_token_length * config.hidden_size ** 2 * 
                                    config.num_key_value_heads/config.num_attention_heads;
        
        // Attention计算量
        const qkFlops = 2 * config.prompt_token_length ** 2 * config.hidden_size;
        const avFlops = 2 * config.prompt_token_length ** 2 * config.hidden_size;
        
        // 输出投影计算量
        const outputProjectionFlops = 2 * config.prompt_token_length * config.hidden_size ** 2;
        
        // 前馈网络计算量
        const ffnFlops = 3 * 2 * config.prompt_token_length * config.hidden_size * config.intermediate_size;
        
        // 每层FLOPS
        const layerFlops = queryProjectionFlops + keyProjectionFlops + valueProjectionFlops + 
                          qkFlops + avFlops + outputProjectionFlops + ffnFlops;
        
        // 总FLOPS
        const totalFlops = layerFlops * config.num_hidden_layers * config.batch_size;
        
        return totalFlops / 1e12; // 转换为TFLOPS
    }
    
    // 计算每个token的解码FLOPS
    function calculateDecodingFLOPsPerToken(config) {
        // Q投影计算量
        const queryProjectionFlops = 2 * config.hidden_size ** 2;
        
        // K,V投影计算量
        const keyProjectionFlops = 2 * config.hidden_size ** 2 * 
                                  config.num_key_value_heads/config.num_attention_heads;
        const valueProjectionFlops = 2 * config.hidden_size ** 2 * 
                                    config.num_key_value_heads/config.num_attention_heads;
        
        // Attention计算量
        const avgSeqLength = config.prompt_token_length + (1 + config.output_token_length) / 2;
        const qkFlops = 2 * avgSeqLength * config.hidden_size;
        const avFlops = 2 * avgSeqLength * config.hidden_size;
        
        // 输出投影计算量
        const outputProjectionFlops = 2 * config.hidden_size ** 2;
        
        // 前馈网络计算量
        const ffnFlops = 3 * 2 * config.hidden_size * config.intermediate_size;
        
        // 每层FLOPS
        const layerFlops = queryProjectionFlops + keyProjectionFlops + valueProjectionFlops + 
                          qkFlops + avFlops + outputProjectionFlops + ffnFlops;
        
        // 总FLOPS
        const totalFlops = layerFlops * config.num_hidden_layers * config.batch_size;
        
        return totalFlops / 1e12; // 转换为TFLOPS
    }
});

