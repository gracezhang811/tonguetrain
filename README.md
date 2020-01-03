# 舌诊
注意事项：   
1.使用keras时出现 `pydot` failed to call GraphViz  
1）因为graphviz需要安装二进制执行文件，所以还需要去官网下一个graphviz安装包安装，并将路径添加到系统环境变量path中  
2）因为pydot已经停止开发了，python3.5以上版本已经用不起来。  
    解决方法是先pip卸载pydot，然后安装pydotplus。  
    pip uninstall pydot  
    pip install pydotplus  
    最后找到keras文件夹下的里面的utils\vis_utils.py打开，把里面的pydot的都替换成pydotplus  
  
