# 定义镜像
training_image = (modal.Image.debian_slim()
    .pip_install([
     
    ])
    .apt_install(
    "git"  # 添加 git 安装
       )
    .run_commands([
        "git clone https://github.com/jingyaogong/minimind /root/minimind",
        "cd /root/minimind && pip install -r requirements.txt",
         "mkdir -p /root/minimind/out"  # 确保输出目录存在
    ])
    .add_local_dir("dataset", remote_path="/root/minimind/dataset")  # 使用add_local_dir
)


