# ✅ (1) 매치 데이터 구조 및 참가자 정보 추출

import json

# match_0.json 파일을 읽어서 파싱 (리그오브레전드 매치 데이터 불러오기)
with open("match_0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))        # 여러 개의 매치가 리스트로 저장돼 있으므로 길이 확인
print(type(data))       # 리스트(list)
single_match = data[0]  # 첫 번째 매치 선택
print(type(single_match))  # 딕셔너리(dict)

# 매치의 메타데이터와 실제 게임 정보(info) 분리
metadata = single_match["metadata"]
print(type(metadata))   # 딕셔너리
info = single_match["info"]
print(type(info))       # 딕셔너리

print(metadata)
print(len(metadata['participants']))  # 참가자 수 = 10명

# 게임 생성 시간 → 사람이 읽을 수 있는 날짜로 변환
import datetime
time = info["gameCreation"]
print(time)             # 밀리초 단위 timestamp
time = time // 1000     # 초 단위로 변환
d = datetime.datetime.fromtimestamp(time, datetime.UTC)
print(d)                # UTC 기준 날짜 출력

# 매치 ID, 생성 시간, 게임 시간 출력
print("*** LoL 매치데이터 정보 ***")
print(f"Match Id : {metadata['matchId']}")
print(f"게임 생성 시간 : {d.strftime('%Y년 %m월 %d일 %H시 %M분')}")
print(f"게임 플레이 시간 : {info['gameDuration']//60}분 {info['gameDuration'] % 60}초")

# 참가자 정보 추출
paticipants = info["participants"]
p = paticipants[0]

# 승리팀 판단: 첫 플레이어 기준으로 팀과 승패 확인
if p["win"]:
    winteam = "블루" if p["teamId"] == 100 else "레드"
else:
    winteam = "레드" if p["teamId"] == 100 else "블루"
print(f"승리 팀 : {winteam}")

# 전체 플레이어 반복 출력
print("\n** 플레이어 정보 **")
for p in paticipants:
    print(f"포지션 : {p['teamPosition']}")  # 탑, 정글 등
    print(f"팀 : {'블루' if p['teamId'] == 100 else '레드'}")
    print(f"챔피언 : {p['championName']}")
    k = p['kills']
    d = p['deaths']
    a = p['assists']
    kda = (k+a) / (1 if d == 0 else d)
    print(f"K/D/A : {k}/{d}/{a}, KDA : {kda:.1f}")
    print(f"챔피언에게 가한 데미지 : {p['totalDamageDealtToChampions']}")
    print(f"받은 데미지 : {p['totalDamageTaken']}")
    print(f"골드 획득량 : {p['goldEarned']}")
    print(f"경험치 획득량 : {p['champExperience']}")
    print()

# ✅ (2) 타임라인 데이터 연동 및 누적 값 추출

# 매치와 타임라인 JSON 파일 로드
with open("match_0.json", "r", encoding="utf-8") as f:
    matches = json.load(f)
with open("timeline_0.json", "r", encoding="utf-8") as f:
    timelines = json.load(f)

# 첫 번째 매치와 타임라인 정보 선택
match    = matches[0]['info']
timeline = timelines[0]['info']

# 첫 번째 플레이어와 참가자 ID 확인
first_player    = match['participants'][0]
first_player_id = first_player['participantId']
print(f"Champion: {first_player['championName']}, pid: {first_player_id}")

# 특정 시간(14분)의 골드 값 추출
minutes = 14
to_extract = 'totalGold'
single_frame = timeline["frames"][minutes]  # 14분 시점의 프레임
target_time = single_frame['timestamp'] / 60000  # 밀리초 → 분

# 참가자의 골드 값 접근
totalGold = single_frame['participantFrames'][str(first_player_id)][to_extract]
print(f"{to_extract} at {target_time:.2f} minutes: {totalGold}")

# ✅ (3) 팀별 골드 집계 및 승패 예측

import json
with open("match_0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

correct_pred = 0
to_extract = 'goldEarned'  # 예측 기준 변수

# 각 매치 반복
for j, match in enumerate(data):
    total_red, total_blue = 0, 0
    info = match['info']
    players = info['participants']

    # 플레이어별로 골드를 팀별 합산
    for p in players:
        score = p[to_extract]
        if p['teamId'] == 100:
            total_blue += score
        else:
            total_red += score

    # 실제 승리 팀 확인
    if players[0]['win']:
        winner = "블루" if players[0]['teamId'] == 100 else "레드"
    else:
        winner = "레드" if players[0]['teamId'] == 100 else "블루"

    # 예측 승리팀 (더 많은 골드를 번 팀)
    pred_winner = "블루" if total_blue > total_red else "레드"
    if winner == pred_winner:
        correct_pred += 1

    print(f"{j + 1:2d}번째 매치 : 승리팀 - {winner}, 예측 - {pred_winner}")
    print(f"{to_extract} : 레드팀 - {total_red}, 블루팀 - {total_blue}\n")

# 전체 예측 정확도 출력
print(f"{to_extract} 변수의 예측 정확도 : {correct_pred/len(data):.2f}")

# ✅ (4) 팀별 누적 경험치 추이 그래프 시각화

# (중간 생략된 파일 로딩 및 기본 구조 동일)
with open('match_0.json','r',encoding='utf-8')as f:
    data_match = json.load(f)
with open('timeline_0.json','r',encoding='utf-8')as f:
    data_timeline = json.load(f)

timeline = data_timeline[0]['info']
match = data_match[0]['info']
players = match['participants']
blue_team = {}
red_team = {}
# 팀별 참가자 ID 저장
for p in players:
    teamId = p['teamId']
    teamPosition = p['teamPosition']
    pid = p['participantId']
    if teamId == 100:
        blue_team[pid] = teamPosition
    else:
        red_team[pid] = teamPosition

# 각 분마다 팀별 경험치 합산
minutes = []
blue_score = []
red_score = []
to_extract = 'xp'  # 경험치

for frame in timeline['frames']:
    time = frame['timestamp'] // 60000
    minutes.append(time)
    blue_score_1min, red_score_1min = 0, 0

    # 각 플레이어의 경험치를 팀별로 더함
    for pid, item in frame['participantFrames'].items():
        if int(pid) in blue_team:
            blue_score_1min += item[to_extract]
        else:
            red_score_1min += item[to_extract]

    blue_score.append(blue_score_1min)
    red_score.append(red_score_1min)

# 시각화
import matplotlib.pyplot as plt

plt.plot(minutes, blue_score, label=f"Blue : {to_extract}", marker='o', linewidth=2)
plt.plot(minutes, red_score, label=f"Red : {to_extract}",  marker='o', linewidth=2)
plt.xlabel('Minutes (m)')
plt.ylabel(f"{to_extract}")
plt.title(f"Feature {to_extract} Graph")
plt.legend()
plt.grid(True)
plt.show()

# 차이값 시각화
diff = []
for j in range(len(minutes)):
    diff.append(blue_score[j] - red_score[j])
plt.figure()
plt.plot(minutes, diff, marker='o', linewidth=2)
plt.show()

# ✅ (5) 특정 포지션만 누적 추이 시각화 (예: MIDDLE 라인)

# 매치 정보와 타임라인 데이터 로드
with open("match_0.json", "r", encoding="utf-8") as f:
    data_match = json.load(f)
with open("timeline_0.json", "r", encoding="utf-8") as f:
    data_timeline = json.load(f)

# info 부분만 추출
timeline = data_timeline[0]['info']
match = data_match[0]['info']

# 참가자 리스트에서 팀과 포지션 정보를 딕셔너리에 저장
players = match['participants']
blue_team = {}
red_team = {}
for p in players:
    teamId = p['teamId']                 # 100: 블루, 200: 레드
    teamPosition = p['teamPosition']     # 포지션: TOP, JUNGLE, MIDDLE 등
    pid = p['participantId']             # 참가자 ID
    if teamId == 100:
        blue_team[pid] = teamPosition
    else:
        red_team[pid] = teamPosition

# 누적 그래프를 위한 리스트 초기화
minutes = []
blue_score = []
red_score = []
to_extract = 'totalGold'    # 누적 골드를 추출
position = 'MIDDLE'         # 분석할 포지션 지정

# 각 시간 프레임에 대해 골드 값을 누적
frames = timeline['frames']
for frame in frames:
    time = frame['timestamp'] // 60000  # 밀리초를 분으로 변환
    minutes.append(time)
    blue_score_1min, red_score_1min = 0, 0
    for pid, item in frame['participantFrames'].items():
        if int(pid) in blue_team:
            if blue_team[int(pid)] == position:       # 블루 팀에서 지정 포지션
                blue_score_1min = item[to_extract]
        else:
            if red_team[int(pid)] == position:        # 레드 팀에서 지정 포지션
                red_score_1min = item[to_extract]
    blue_score.append(blue_score_1min)
    red_score.append(red_score_1min)

# 그래프 시각화
import matplotlib.pyplot as plt

# 팀별 누적 골드 추이 그래프
plt.plot(minutes, blue_score, label=f"Blue[{position}] : {to_extract}",
         marker='o', linewidth=2)
plt.plot(minutes, red_score, label=f"Red[{position}] : {to_extract}",
         marker='o', linewidth=2)

plt.xlabel('Minutes (m)')
plt.ylabel(f"{to_extract}")
plt.title(f"Feature {to_extract} Graph, Position = {position}")
plt.legend()
plt.grid(True)
plt.show()

# 블루-레드 골드 차이 시각화
plt.figure()
diff = []
for j in range(len(minutes)):
    diff.append(blue_score[j] - red_score[j])
plt.plot(minutes, diff, marker='o', linewidth=2)
plt.show()
