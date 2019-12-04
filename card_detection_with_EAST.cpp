#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <algorithm>

#define inpWidth 640
#define inpHeight  480
#define confThreshold 0.5
#define nmsThreshold  0.4

using namespace cv;
using namespace cv::dnn;
using namespace std;

void east_process(Mat target);
void decode(const Mat& scores, const Mat& geometry, float scoreThresh, vector<RotatedRect>& detections, vector<float>& confidences);
void make_target();
Mat target; // 결과 이미지 저장용
Mat roi;

int main()
{
	while (1)
	{
		make_target();
		try
		{
			east_process(target);
		}
		catch(...)
		{
			cout << "error\n";
			destroyAllWindows();
		}
	}
	
	return 0;
}

void make_target()
{
	VideoCapture cap(0);

	//cap.set(CAP_PROP_FRAME_WIDTH, inpWidth);
	//cap.set(CAP_PROP_FRAME_HEIGHT, inpHeight);

	if (!cap.isOpened())
	{
		cerr << "camera open err.\n";
		exit(-1);
	}

	Mat frame;
	while (1)
	{
		// 카메라로 촬영 이미지를 frame에 저장
		cap >> frame;
		if (frame.empty()) {
			cerr << "frame is empty!.\n";
			break;
		}

		imshow("web cam", frame);

		if (waitKey(25) == 115) // 's'의 아스키 코드값이 115
		{
			target = frame.clone();
			roi = frame.clone();
			destroyAllWindows();
			break;
		}
	}
}


void east_process(Mat target)
{

	String model = "C:/Users/kwon/Desktop/cv_4/frozen_east_text_detection.pb";

	CV_Assert(!model.empty());

	// Load network.
	Net net = readNet(model);


	static const std::string kWinName = "EAST RESULT"; // 윈도우 이름 설정해서 생성
	namedWindow(kWinName, WINDOW_NORMAL);

	std::vector<Mat> outs;
	std::vector<String> outNames(2);
	outNames[0] = "feature_fusion/Conv_7/Sigmoid";
	outNames[1] = "feature_fusion/concat_3";

	Mat blob;

	blobFromImage(target, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false); // deep learning
	net.setInput(blob);
	net.forward(outs, outNames);

	Mat scores = outs[0];
	Mat geometry = outs[1];

	// Decode predicted bounding boxes.
	std::vector<RotatedRect> boxes;
	std::vector<float> confidences;
	decode(scores, geometry, confThreshold, boxes, confidences);

	// Apply non-maximum suppression procedure.
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Render detections.
	Point2f ratio((float)target.cols / inpWidth, (float)target.rows / inpHeight);

	Point2f * top_left = new Point2f[indices.size()];
	Point2f * bottom_right = new Point2f[indices.size()];
	int* index = new int[indices.size()];
	int* temp_index = new int[indices.size()];

	for (size_t i = 0; i < indices.size(); ++i)
	{
		
		RotatedRect& box = boxes[indices[i]];

		Point2f vertices[4];
		box.points(vertices);

		for (int j = 0; j < 4; ++j)
		{
			vertices[j].x *= ratio.x;
			vertices[j].y *= ratio.y;
		}
		index[i] = i;
		top_left[i] = vertices[1];
		bottom_right[i] = vertices[3];
	}

	float max_x = 0;
	float min_x = 0;
	float max_y = 0;
	float min_y = 0;
	int cnt = 0;
	int t_i = 0;
	int last_index = 0;
	for (size_t i = 0; i < indices.size(); ++i) // 자기 자신
	{
		RotatedRect& box = boxes[indices[i]];

		Point2f vertices[4];
		box.points(vertices);
		for (int k = 0; k < 4; ++k)
		{
			vertices[k].x *= ratio.x;
			vertices[k].y *= ratio.y;
		}

		for (size_t j = 0; j < indices.size(); ++j) // 비교 대상
		{

			if (abs(top_left[i].y - top_left[j].y) <= 5 && i != j)
			{
				cnt++;
				temp_index[t_i] = i;
				t_i++;
			}
		}

		if (cnt >= 2) // 자신 제외하고 비슷한 y가  2개이상 있을때!
		{
			//for (int j = 0; j < 4; ++j)   // vertices[0] : bottom_left  , vertices[1] : top_left, vertices[2] : top_right, vertices[3] : bottom_right
			//{
			//	line(target, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 1);
			//}
			

			min_x = top_left[0].x;
			min_y = top_left[0].y;
			for (size_t j = 0; j < indices.size(); ++j)
			{
				if (max_x < bottom_right[j].x) // 비슷한 박스중 맨 오른쪽 x
				{
					max_x = bottom_right[j].x;
					max_y = bottom_right[j].y;
				}
				if (min_x > top_left[j].x)  // 비슷한 박스중 맨 왼쪽 x
				{
					min_x = top_left[j].x;
					min_y = top_left[j].y;
				}
			}
		}

		cnt = 0;
		t_i = 0;
	}

	//rectangle(target, Point(min_x, min_y), Point(max_x, max_y), Scalar(255, 0, 0), 2);

	int w = max_x - min_x;
	int h = max_y - min_y;

	
	roi = target(Rect(min_x - 10, min_y - 10, w + 20, h + 20));
	if (roi.empty())
		return;
	imshow("roi",roi);
	waitKey();


	// Put efficiency information.
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(target, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));


	imshow(kWinName, target);
	waitKey();
	destroyAllWindows();
	delete[] top_left;
	delete[] index;
	delete[] temp_index;

	cv::Size sz = target.size();
	int imageWidth = sz.width;
	int imageHeight = sz.height;

}


void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
	std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
	detections.clear();
	CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		const float* scoresData = scores.ptr<float>(0, 0, y);
		const float* x0_data = geometry.ptr<float>(0, 0, y);
		const float* x1_data = geometry.ptr<float>(0, 1, y);
		const float* x2_data = geometry.ptr<float>(0, 2, y);
		const float* x3_data = geometry.ptr<float>(0, 3, y);
		const float* anglesData = geometry.ptr<float>(0, 4, y);
		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			if (score < scoreThresh)
				continue;

			// Decode a prediction.
			// Multiple by 4 because feature maps are 4 time less than input image.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];

			Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
				offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
			Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
			RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			detections.push_back(r);
			confidences.push_back(score);
		}
	}
}