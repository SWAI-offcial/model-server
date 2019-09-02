## 目录结构
#### modelServer
```
|---modelServer
|------app.py 项目启动文件
|------requirements.txt 项目配置文件
|------models 模型文件目录
|---------modelName 某个模型（自定义文件夹名称，按照驼峰式命名） 
|------assets 资源目录（模型以外的资源文件）
|------.gitignore 上传时需要忽略的文件（请再三检查！！！）
|------readme.md 说明文件（添加新项目或者修改就像幕后请添加修改情况！！！）
|------deploy.bash 项目发布脚本
```