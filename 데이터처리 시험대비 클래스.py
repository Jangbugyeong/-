#클래스는 함수+데이터
#사칙연산을 하는 클래스 만들기
#set_data라는 함수를 만들어서 계산을 하기 전에 값을 세팅하기
class Calc:
    def __init__(self):
        pass

    def add(self,x,y):
        self.x=x
        self.y=y
        self.result = self.x+self.y
        return self.result
    def mul(self,x,y):
        self.x=x
        self.y=y
        self.result = self.x*self.y
        return self.result
    def sub(self,x,y):
        return self.add(x,-y)
    def div(self,x,y):
        return self.mul(x,1.0/y)

class Calc2:
    def __init__(self):
        self.x=1
        self.y=1
    def set_data(self,x,y):
        self.x=x
        self.y=y

    def add(self):
        return self.x+self.y
    def sub(self):
        return self.x-self.y
    def mul(self):
        return self.x*self.y
    def div(self):
        return self.x/self.y
c= Calc()
print(c.add(3,4))
print(c.x,c.y,c.result)
print(c.mul(5,6))
print(c.x,c.y,c.result)
print(c.sub(7,8))
print(c.x,c.y,c.result)
print(c.div(9,10))
print(c.x,c.y,c.result)

d=Calc()
print(d.add(10,20))

e=Calc2()
e.set_data(3,4)
print(e.add())
print(e.mul())
print(e.div())
print(e.sub())


