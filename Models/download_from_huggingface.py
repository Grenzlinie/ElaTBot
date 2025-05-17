import os
import argparse
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

def download_model(model_name, save_path, model_type="all", use_auth_token=None):
    """
    从Hugging Face下载模型到本地指定文件夹
    
    参数:
    model_name -- Hugging Face上的模型名称，例如："bert-base-uncased"
    save_path -- 本地保存路径
    model_type -- 下载类型，可选："all"(完整模型)，"weights"(仅权重)，"tokenizer"(仅分词器)
    use_auth_token -- Hugging Face的认证token，用于访问私有模型或Gate模型
    
    返回:
    下载的模型路径
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 完整的保存路径（模型名称作为子文件夹）
    model_path = os.path.join(save_path, model_name.split("/")[-1])
    os.makedirs(model_path, exist_ok=True)
    
    print(f"开始下载模型 {model_name} 到 {model_path}...")
    
    try:
        if model_type == "all" or model_type == "weights":
            # 使用snapshot_download下载完整模型
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                use_auth_token=use_auth_token
            )
            print(f"模型权重已下载到 {model_path}")
        
        if model_type == "all" or model_type == "tokenizer":
            # 下载并保存tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
            tokenizer.save_pretrained(model_path)
            print(f"Tokenizer已下载到 {model_path}")
            
        print(f"模型 {model_name} 已成功下载到 {model_path}")
        return model_path
        
    except Exception as e:
        print(f"下载过程中出现错误: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从Hugging Face下载模型到本地")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face上的模型名称")
    parser.add_argument("--save_path", type=str, required=True, help="本地保存路径")
    parser.add_argument("--model_type", type=str, default="all", choices=["all", "weights", "tokenizer"], help="下载类型")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face的认证token")
    
    args = parser.parse_args()
    
    download_model(
        model_name=args.model_name,
        save_path=args.save_path,
        model_type=args.model_type,
        use_auth_token=args.token
    )