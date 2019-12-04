# credit_card_number_detection
with OpenCV

## 1. 개요

### 1) 카메라로 카드번호를 추출하는 프로그램 (roi_save.cpp)
### 2) 추출한 카드번호를 선명하게 만들어주는 프로그램 (sharpening.cpp)  
### 3) 카드 번호 추출 deep learning version (card_detection_with_EAST.cpp) 

## 2. 구현 환경

1) window10 Home
2) c++
3) OpenCV 4.1.0 ~ 4.1.1

## 3. 동작 과정

### ㄱ) roi_save.cpp

1) 이미지 resize(), bilateralFilter() , morphologyEx() 등을 통해 전처리
2) 가로, 세로, 종횡비로 한번 거르고, 추출한 roi에서 넓이를 기준으로 후보를 선정 (3개까지)
3) warpPerspective()적용하여, 위치 보정 후 출력


### ㄴ) sharpening.cpp

1) roi_save.cpp 와 동일한 과정을 거쳐 roi추출
2) 추출한 roi에 addWeighted(), bilateralFilter() 등 선명도가 최선인 경우를 탐색

### ㄷ) card_detection_with_EAST.cpp

1) 카메라로 이미지 촬영 후 target 이미지로 선택
2) target 이미지를 east알고리즘에 돌려 모든 text영역 추출
3) 추출된 text영역 중에서 비슷한 y좌표를 가진 영역들만 선택
4) top_left와 bottom_rigthtf를 roi로 추출


## roi_save.cpp 예시
## 원본)
![card](https://user-images.githubusercontent.com/46870741/66713961-7a82ca00-edeb-11e9-8b7e-f146ff5bd79d.png)

## 추출)
![roi](https://user-images.githubusercontent.com/46870741/66713985-ccc3eb00-edeb-11e9-9bb6-6fbf16d55ef7.png)


## sharpening.cpp 예시

## 선명도)
![sharp](https://user-images.githubusercontent.com/46870741/66713989-d64d5300-edeb-11e9-94db-be35dd4ee5a3.png)

## card_detection_with_EAST.cpp 예시

## target 이미지 및 걸린 시간)
![EAST RESULT_1](https://user-images.githubusercontent.com/46870741/70132727-20560500-16c8-11ea-9a7b-de4cc47c59e1.png)

## 
![east_result](https://user-images.githubusercontent.com/46870741/70132726-20560500-16c8-11ea-82ac-d4d5c7afb0fd.png)

