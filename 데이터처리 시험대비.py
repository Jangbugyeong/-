#%d는 정수
#%s 문자열
#%f 부동소수점
#%% % 기호 자체를 의미

#append = 원소 추가
#len = 길이 확인
#del 변수[리스트 원소] = 리스트 원소 삭제
#sort = 정렬(오름차순)
#sort(reverse=True) = 정렬(내림차순)

#리스트[]
#문자열과 유사하다 (인덱싱,슬라이싱)

#튜플()
#더할 때 더하는 두 개의 대상이 모두 튜플이어야 한다.
#튜플 원소 삭제 불가능

#딕셔너리(json과 비슷):{}중괄호 사용
#key:value의 형태 예) a:100일때 a=key 100=value
#인덱싱이 불가능
#value에는 리스트가 올 수 있음
#추가하려면 딕셔너리이름['키값'] = value
#key에 대한 value는 하나

#집합 자료형 만들기
s1 = set([1, 2, 3])
s2 = set("Hello")

#집합 자료형의 특징
#중복을 허용하지 않는다.
#순서가 없다(Unordered).
#만약 set 자료형에 저장된 값을 인덱싱으로 접근하려면 다음과 같이 리스트나 튜플로 변환한 후에 해야 한다.
l1 = list(s1)
l1[0]
t1 = tuple(s1)
t1[0]

s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])

#교집합,합집합,차집합 구하기
s1 & s2 # 교집합
s1 | s2 # 합잡합
s1 - s2 #차집합 {1, 2, 3}
s2 - s1 #{8, 9, 7}

#값 1개 추가하기
s1 = set([1, 2, 3])
s1.add(4)
#값 여러 개 추가하기 - update
s1 = set([1, 2, 3])
s1.update([4, 5, 6])
#특정 값 제거하기 - remove
s1 = set([1, 2, 3])
s1.remove(2)



#비교연산자
# a==b : a와 b가 같은가, a!=b : a와 b가 같지 않은가
#문자열의 경우 빈 문자열이면 False를 반환,문자가 하나라도 있으면 True
#숫자의 경우 0이면 False를 반환,아니라면 True를 반환

#리스트,튜플,딕셔너리
#비어있다면 False,무엇이라도 있다면 True

#None이라면 무엇이든 False



money=4000
if (money >= 3000) and (money <= 5000):
    print('버스')
else:
    print('걸어가자')


#if 조건문:
    #수행할_명령1
#elif 조건문:
    #수행할_명령 11
#else:
    #수행할_명령a (else문은 없어도 가능)
#비교연산자의 경우 조건이 맞으면 True

# and
#True and True == True
#True and False == False
#False and True == False
#False and False == False

# or
#True and True == True
#True and False == True
#False and True == True
#False and False == False

# not
#not true == False
#not False == True

#in == 안에 있으면 True
#not in == 안에 없으면 True

pocket = ['phone','paper','card']

if 'money' in pocket:
    print('택시')
elif 'card' in pocket:
    print('버스')
else:
    print('걸어가자')

#for 변수 in 리스트(튜플,문자열):
#    변수가 순서대로 변화

score= [90,25,60,75,85]
# 80점 이상이면 합격이고 아니면 탈락이라고 알려주는 프로그램

for s in score:
    if s >=80:
        print('합격,점수는 %d점 입니다.'% s)
    else:
        print('탈락,점수는 %d점 입니다.'% s)

# while문 예제 실습
# 기본 형태
#while 조건문:
    #수행할_문장1
    #수행할_문장2
    #수행할_문장3
treeHit = 0
while treeHit < 10:
    treeHit = treeHit +1
    print("나무를 %d번 찍었습니다." % treeHit)
    if treeHit == 10:
        print("나무 넘어갑니다.")

prompt = """
1. Add
2. Del
3. List
4. Quit

Enter number: """

number = 0
while number != 4:
    print(prompt)
    number = int(input())

coffee = 10
money = 300
while money:
    print("돈을 받았으니 커피를 줍니다.")
    coffee = coffee - 1
    print("남은 커피의 양은 %d개입니다." % coffee)
    if coffee == 0:
        print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
    break

coffee = 10
while True:
    money = int(input("돈을 넣어 주세요: "))
    if money == 300:
        print("커피를 줍니다.")
        coffee = coffee - 1
    elif money > 300:
        print("거스름돈 %d를 주고 커피를 줍니다." % (money - 300))
        coffee = coffee - 1
    else:
        print("돈을 다시 돌려주고 커피를 주지 않습니다.")
        print("남은 커피의 양은 %d개 입니다." % coffee)
    if coffee == 0:
        print("커피가 다 떨어졌습니다. 판매를 중지 합니다.")
        break

a = 0
while a < 10:
    a = a + 1
    if a % 2 == 0: continue
    print(a)
#함수의 문법 이해

#def 함수이름(함수에 대한 입력값(없을 수도 있음)):
    #return 반환하는 값

#입력과 반환값이 모두 있는 전형적인 함수
def add (a,b):
    return a+b
#입력값과 반환값이 모두 없는 함수
def say():
    print('안녕')
#입력값은 없으나 반환값이 있는 함수
def say_str():
    return '안녕_say_str'
#입력값은 있으나 반환값이 없는 함수
def say_add(a,b):
    print("%d와 %d의 합은 %d입니다."% (a,b,a+b))

def calc(a,b,opt='add'):
    if opt == 'add':
        return  a+b
    elif opt == 'mul':
        return a*b
    else:
        print('calc 함수의 opt는 add나 mul이 되어야 합니다.')
        return None

def add_num(num):
    sum=0
    for j in range(1,num+1): # range(시작,끝,몇칸씩)
        sum+=j
    return sum

def print_triangle(num_star):
    for j in range(1,num_star+1):
        print(j*'*')
    print('*')

def print_diamond(num_diamond):
    #별의 수는 1,3,5,3,1 num_star
    #빈 공간의 수는 2,1,0,1,2 num_space
    star = num_diamond // 2
    num_space = []
    num_star = []
    for j in range(-star,star+1):
        num_space.append(abs(j))
    for s in num_space:
        num_star.append(num_diamond-s*2)
    for space,star in zip(num_space,num_star):
        print(space*' '+star * '*')

c = add(3,4)
print(c)
say()
d=say_str()
print(d)
say_add(3,5)
print(calc(3,4,'add'))
print(calc(3,4,'mul'))
print(calc(3,4,'aaa'))
print(calc(3,4))
sum = add_num(10)

num_star= 10
print_triangle(num_star)
num_diamond =9
print_diamond(num_diamond)



