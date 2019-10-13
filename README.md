# credit_card_number_detection
with OpenCV

## 1. 개요

1) 카메라로 카드번호를 추출하는 프로그램 (roi_save.cpp)
2) 추출한 카드번호를 선명하게 만들어주는 프로그램 (sharpening.cpp)

## 2. 구현 환경

1) window10 Home
2) c++
3) OpenCV 4.1.0

## 3. 동작 과정

ㄱ) roi_save.cpp

1) 이미지 resize(), bilateralFilter() , morphologyEx() 등을 통해 전처리
2) 가로, 세로, 종횡비로 한번 거르고, 추출한 roi에서 넓이를 기준으로 후보를 선정 (3개까지)
3) warpPerspective()적용하여, 위치 보정 후 출력


## 예시
원본)
![card](https://user-images.githubusercontent.com/46870741/66713961-7a82ca00-edeb-11e9-8b7e-f146ff5bd79d.png)
추출)
![roi](https://user-images.githubusercontent.com/46870741/66713985-ccc3eb00-edeb-11e9-9bb6-6fbf16d55ef7.png)


ㄴ) sharpening.cpp

1) roi_save.cpp 와 동일한 과정을 거쳐 roi추출
2) 추출한 roi에 addWeighted(), bilateralFilter() 등 선명도가 최선인 경우를 탐색

## 예시
선명도)
![sharp](https://user-images.githubusercontent.com/46870741/66713989-d64d5300-edeb-11e9-94db-be35dd4ee5a3.png)
