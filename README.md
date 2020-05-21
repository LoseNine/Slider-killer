# Slider-killer
滑块验证码杀手，识别率99%，输入路径返回坐标

环境pip install
```python
tensorflow-gpu     1.11.0
numpy              1.18.4
opencv-python      4.2.0.34
```

代码实例
```python
from core.getLoc import getPageLoc

if __name__=="__main__":
	result=getPageLoc("./show/yd.jpg")
	print(result)

```

易盾测试
![](https://github.com/LoseNine/Slider-killer/blob/master/show/show.PNG)
