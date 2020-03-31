// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <queue> 
#include <random>


using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
////////////////////////////////////////////////// LAB  1 ////////////////////////////////////////////////////////////////
void negative_image() {
	Mat img = imread("Images/kids.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
	imshow("negative image", img);
	waitKey(0);
}

void additive_factor(int factor) {
	Mat img = imread("Images/kids.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int value = img.at<uchar>(i, j) + factor;
			if (value < 255 && value > 0) 
				img.at<uchar>(i, j) += factor;
			else if (value > 255) 
				img.at<uchar>(i, j) = 255;
			else 
				img.at<uchar>(i, j) = 0;
		}
	}
	imshow("additive image", img);
	waitKey(0);
}

void multiplicative_factor(int factor) {
	Mat img = imread("Images/kids.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int value = img.at<uchar>(i, j) * factor;
			if (value < 255 && value > 0)
				img.at<uchar>(i, j) *= factor;
			else if (value > 255)
				img.at<uchar>(i, j) = 255;
			else
				img.at<uchar>(i, j) = 0;
		}
	}
	imshow("multiplicative image", img);

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	imwrite("Images/alpha.png", img, compression_params);

	waitKey(0);
}


Mat show_initial_image_grayscale(String path) {
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("initial image", img);
	return img;
}

void create4squares_matrix() {
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (i < 127 && j < 127) {
				Vec3b pixel;
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255;
				img.at<Vec3b>(i, j) = pixel;
			}
			if (i < 127 && j > 127) {
				Vec3b pixel;
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 255;
				img.at<Vec3b>(i, j) = pixel;
			}
			if (i > 127 && j < 127) {
				Vec3b pixel;
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 0;
				img.at<Vec3b>(i, j) = pixel;
			}
			if (i > 127 && j > 127) {
				Vec3b pixel;
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 255;
				img.at<Vec3b>(i, j) = pixel;
			}
		}
	}
	imshow("squares image", img);
	waitKey(0);
}

void inverse() {
	Mat img(3, 3, CV_8UC3);
	img.inv();
}


////////////////////////////////////////////////////// LAB 2 /////////////////////////////////////////////////////////////

void color_to_greyscale_components(Mat img) {
	CV_Assert(img.type() == CV_8UC3);
	Mat blue_grayscale(img.rows, img.cols, CV_8UC1);
	Mat green_grayscale(img.rows, img.cols, CV_8UC1);
	Mat red_grayscale(img.rows, img.cols, CV_8UC1);
	Vec3b pixel;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			pixel                           = img.at<Vec3b>(i, j);
			blue_grayscale.at<uchar>(i, j)  = pixel[0];
			green_grayscale.at<uchar>(i, j) = pixel[1];
			red_grayscale.at<uchar>(i, j)   = pixel[2];
		}
	}
	imshow("color image", img);
	imshow("blue component", blue_grayscale);
	imshow("green component", green_grayscale);
	imshow("red component", red_grayscale);
	waitKey(0);

}

void color_to_greyscale(Mat img) {
	CV_Assert(img.type() == CV_8UC3);
	Mat grayscale(img.rows, img.cols, CV_8UC1);
	Vec3b pixel;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			pixel                     = img.at<Vec3b>(i, j);
			grayscale.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
		}
	}
	imshow("color image", img);
	imshow("grayscale image", grayscale);
	waitKey(0);
}

Mat black_white;
Mat grayscale_to_black_and_white(Mat img, int threshold){
	CV_Assert(img.type() == CV_8UC1);
	black_white = img;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < threshold) black_white.at<uchar>(i, j) = 0;
			else black_white.at<uchar>(i, j) = 255;
		}
	}
	imshow("grayscale image", img);
	imshow("black and white image", black_white);
	return black_white;
	waitKey(0);
}

void rgb_to_hsv() {
	
}

void isInside(Mat img, int i, int j) {
	CV_Assert(i < img.rows && j < img.cols);
}


bool isInside2(Mat img, int i, int j) {
	if (i < img.rows && i >= 0 && j >= 0 && j < img.cols) return true;
	return false;
}

void yellow_detection() {
	char fname[MAX_PATH];
	float r, g, b;
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat yellow_detection(img.rows, img.cols, CV_8UC1);
		Mat hue_grayscale(img.rows, img.cols, CV_8UC1);
		Mat saturation_grayscale(img.rows, img.cols, CV_8UC1);
		Mat value_grayscale(img.rows, img.cols, CV_8UC1);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				Vec3b pixel = img.at<Vec3b>(i, j);
				b = ((float)pixel[0]) / 255.0f;
				g = ((float)pixel[1]) / 255.0f;
				r = ((float)pixel[2]) / 255.0f;

				float M = max(max(r, g), b);
				float m = min(min(r, g), b);
				float C = M - m;

				// Value computation
				value_grayscale.at<uchar>(i, j) = (uchar)(255 * M);

				// Saturation computation
				if (value_grayscale.at<uchar>(i, j) != 0)
					saturation_grayscale.at<uchar>(i, j) = (uchar)((255 * C) / M);
				else saturation_grayscale.at<uchar>(i, j) = 0;

				// Hue computation
				float hue;
				if (C != 0) {
					if (M == r) hue = 60.0f * (g - b) / C;
					if (M == g) hue = 120.0f + 60.0f * (b - r) / C;
					if (M == b) hue = 240.0f + 60.0f * (r - g) / C;
				}
				else hue = 0.0f;
				if (hue < 0) hue += 360.0f;
				hue_grayscale.at<uchar>(i, j) = (uchar)(hue * 255.0f / 360.0f);
				if (hue > 40.0f && hue < 80.0f) {
					if (saturation_grayscale.at<uchar>(i, j) * 100 / 255 > 50.0f) {
						yellow_detection.at<uchar>(i, j) = 255;
					}
					else yellow_detection.at<uchar>(i, j) = 0;
				}
				else yellow_detection.at<uchar>(i, j) = 0;
			}
		}
		imshow("image", img);
		imshow("yellow detection", yellow_detection);
		waitKey(0);
	}
}

///////////////////////////////////////// LAB 3 //////////////////////////////////////////////////////////////////

int g[256];

int* histogram( char * f = NULL) {
	char fname[MAX_PATH];
	for (int z = 0; z < 255; z++)
		g[z] = 0;
	if (f == NULL) {
		while (openFileDlg(fname)) {
			Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			
			//for (int k = 0; k < 255; k++) {
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					g[img.at<uchar>(i, j)]++;
				}
			}
			//}
			imshow("image", img);
			showHistogram("histogram", g, img.rows, img.cols);
			waitKey(0);
		}
	}
	else {
		
		Mat img = imread(f, CV_LOAD_IMAGE_GRAYSCALE);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				g[img.at<uchar>(i, j)]++;				}
		}
	
		imshow("image", img);
		showHistogram("histogram", g, img.rows, img.cols);
		waitKey(0);
		
	}
	return g;
}
float p[256];

float* pdf() {
	int* g;
	for (int z = 0; z < 255; z++)
		p[z] = 0;
	g = histogram("Images/cameraman.bmp");
	for (int i = 0; i < 255; i++) {
		p[i] = (float)g[i] / (float)((imread("Images/cameraman.bmp", CV_LOAD_IMAGE_UNCHANGED).rows)
				* (float)(imread("Images/cameraman.bmp", CV_LOAD_IMAGE_UNCHANGED).cols));
	}
	imshow("image", imread("Images/cameraman.bmp", CV_LOAD_IMAGE_UNCHANGED));
	showHistogram("pdf", (int*)p, imread("Images/cameraman.bmp", CV_LOAD_IMAGE_UNCHANGED).rows, imread("Images/cameraman.bmp", CV_LOAD_IMAGE_UNCHANGED).cols);
	
	waitKey(0);
	return p;
}

std::vector<int> threshold_values;

std::vector<int> multilevel_thresholding(float * p) {
	int wh = 5;
	threshold_values.push_back(0);
	float th = 0.0003f;
	for (int i = 0 + wh; i < 255 - wh + 1; i++) {
		float v = 0.0f;
		float max_value = 0.0f;
		for (int k = i - wh; k <= i + wh; k++) {
			v += p[k];
			if (p[k] > max_value)
				max_value = p[k];
		}
		v /= (2 * wh + 1);
		
			if (p[i] > v + th && p[i] == max_value) {
				threshold_values.push_back(i);
			}
		
	}
	threshold_values.push_back(255);
	return threshold_values;
}


void print_threshold_image(std::vector<int> a) {
	int length = a.size() + 1;
	int ab[15], k=0;
	for (std::vector<int>::const_iterator i = a.begin(); i != a.end(); ++i) {
		ab[k] = *i;
		k++;
	}
	int j = 0;
	int cur_value = ab[0] , next_value = ab[1];
	int values[256];
	for (int i = 0; i < 255; i++) {
		values[i] = cur_value;
		if (next_value - i < i - cur_value) {
			j++;
			cur_value = ab[j];
			next_value = ab[j + 1];
		}
	}
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("image", img);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = (uchar)values[img.at<uchar>(i, j)];
		}
	}
	imshow("image2", img);
	waitKey(0);
}

void floyd_steinberg() {

}

///////////////////////////////////////////////////////////////  LAB 4  //////////////////////////////////////////////////////////////////////////////////////////

int check_if_in_object(Mat img,int row, int col, int greyscale_value = 0) {
	if (img.at<uchar>(row, col) == greyscale_value) 
		return 1;
	else return 0;
}

int area_val;
int area(Mat img, int label) {
	area_val = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (check_if_in_object(img, i, j, label))
				area_val++;
		}
	}
	return area_val;
}

int total_rows, total_cols;
int get_I_COM(Mat img, int label) {
	total_rows = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			total_rows += i * check_if_in_object(img, i, j, label);
		}
	}
	return total_rows / area(img, label) ;
}

int get_J_COM(Mat img, int label) {
	total_cols = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			total_cols += j * check_if_in_object(img, i, j, label);
		}
	}
	return  total_cols / area(img, label);
}

float axis_of_elongation(Mat img, int label) {
	float nominator = 0, denominator1 = 0, denominator2 = 0;
	int i_COM = get_I_COM(img, label);
	int j_COM = get_J_COM(img, label);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			nominator += (i - i_COM) *
						 (j - j_COM) *
						 check_if_in_object(img, i, j, label);

			denominator1 += (j - j_COM) *
						    (j - j_COM) *
						    check_if_in_object(img, i, j, label);

			denominator2 += (i - i_COM) *
						    (i - i_COM) *
						    check_if_in_object(img, i, j, label);
							
		}
	}
	float teta = (atan2(2 * nominator, (denominator1 - denominator2)))/2;
	return teta;
}

int a[8] = { 1, 1,  1, 0,  0, -1, -1, -1 };
int b[8] = { 1, 0, -1, 1, -1,  1,  0, -1 };
float per;
int perimeter(Mat img, int label = 0) {
	per = 0;
	bool found = false;
	Mat buff1(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (check_if_in_object(img, i, j, 0) == 1) {
				for (int k = 0; k < 8; k++) {
					if (check_if_in_object(img, i + a[k], j + b[k], 0) == 0)
						found = true;
					else buff1.at<uchar>(i, j) = 255;
				}
				if (found) {
					per++;
					buff1.at<uchar>(i, j) = 0;
					found = false;
				}
			}
			else buff1.at<uchar>(i, j) = 255;
		}
	}
	imshow("Contour", buff1);
	return per * atan(1);
}

float thiness_ratio(Mat img, int label = 0) { //atan 1 * 4 e pi
	return 4.0f * 
		atan(1) * 4.0f * 
		((float)area(img, label) / 
		((float)perimeter(img, label) * (float)perimeter(img, label)));
}

int row_min(Mat img, int label=0) {
	bool first = false;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (first == false && check_if_in_object(img, i, j, label) == 1)
			{
				first = true;
				return i;
			}
		}
	}
}


int row_max(Mat img, int label=0) {
	bool first = false;
	for (int i = img.rows - 1 ; i >= 0; i--) {
		for (int j = 0; j < img.cols; j++) {
			if (first == false && check_if_in_object(img, i, j, label) == 1)
			{
				first = true;
				return i;
			}
		}
	}
}


int col_min(Mat img, int label=0) {
	bool first = false;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			if (first == false && check_if_in_object(img, j, i, label) == 1)
			{
				first = true;
				return i;
			}
		}
	}
}


int col_max(Mat img, int label=0) {
	bool first = false;
	for (int i = img.cols - 1; i >= 0 ; i--) {
		for (int j = 0; j < img.rows; j++) {
			if (first == false && check_if_in_object(img, j, i, label) == 1)
			{
				first = true;
				return i;
			}
		}
	}
}

int * horiz;
void projection_horizontal(Mat img, int label = 0) {
	int i = 0;
	horiz = (int*)malloc((img.rows + 1)* sizeof(int));
	memset(horiz, 0, img.rows*4);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			horiz[i] += check_if_in_object(img, i, j);
		}
	}
	//return horiz;
}


int * vert;
void  projection_vertical(Mat img, int label = 0) {
	int i = 0;
	vert = (int*)malloc((img.cols + 1) * sizeof(int));
	memset(vert, 0, img.cols*4);
	for (int j = 0; j < img.cols; j++) {
		for (int i = 0; i < img.rows; i++) {
			vert[j] += check_if_in_object(img, i, j);
		}
	}
	//return vert;
}

////////////////////////////////////////////////////////////////////////////////// LAB 5 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct coords {
	int i;
	int j;
}coord;

Mat labels = imread("Images/letters.bmp",CV_LOAD_IMAGE_GRAYSCALE);
Mat breadth_first_labelling(Mat img) {
	short label = 0;
	int x = img.rows, y = img.cols;
	labels.zeros(labels.rows, labels.cols, 1);

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
				label++;
				queue <coord> Q;
				labels.at<uchar>(i,j) = label;
				coord current;
				current.i = i;
				current.j = j;
				Q.push(current);
				while (Q.size() > 0)
				{
					coord popped = Q.front();
					Q.pop();
					for (int k = 0; k < 8; k++) {
						if (isInside2(img, popped.i + a[k], popped.j + b[k])) {
							if (img.at<uchar>(popped.i + a[k], popped.j + b[k]) == 0 && labels.at<uchar>(popped.i + a[k], popped.j + b[k]) == 0) {
								labels.at<uchar>(popped.i + a[k], popped.j + b[k]) = label;
								coord neigh;
								neigh.i = popped.i + a[k];
								neigh.j = popped.j + b[k];
								Q.push(neigh);
							}
						}
					}
				}
			}
		}
	}
	printf("\n\n %d \n\n", label);
	return labels;
}

int main()
{
	int factor, x = 0, y = 0;
	Point pStart;
	Point pEnd;	
	int length = 70;
	Mat img1 = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img, buff1(img1.rows, img1.cols, CV_8UC3), buff2(img1.rows, img1.cols, CV_8UC3);
	Mat buff;
	int r, g, b1;
	Mat labelled;
	uchar label = 0;
	int maybe = 0;
	uchar used[1000];
	std::vector<int> a1;
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1  - Open image\n");
		printf(" 2  - Open BMP images from folder\n");
		printf(" 3  - Image negative - diblook style\n");
		printf(" 4  - BGR->HSV\n");
		printf(" 5  - Resize image\n");
		printf(" 6  - Canny edge detection\n");
		printf(" 7  - Edges in a video sequence\n");
		printf(" 8  - Snap frame from live video\n");
		printf(" 9  - Mouse callback demo\n");
		printf(" 10 - My negative image\n");
		printf(" 11 - My additive factor\n");
		printf(" 12 - My multiplicative factor\n");
		printf(" 13 - My 4 squares\n");
		printf(" 14 - My inverse\n");
		printf(" 15 - Color image to grayscale components\n");
		printf(" 16 - Color image to grayscale\n");
		printf(" 17 - Grayscale image to black and white\n");
		printf(" 18 - My RGB to HSV\n");
		printf(" 19 - Test if point in image\n");
		printf(" 20 - Histogram\n");
		printf(" 21 - Yellow detection\n");
		printf(" 22 - PDF\n");
		printf(" 23 - Multilevel thresholding\n");
		printf(" 24 - Check if inside\n");
		printf(" 25 - Test for area\n");
		printf(" 26 - The center of mass\n");
		printf(" 27 - Print axis of elongation\n");
		printf(" 28 - Perimeter\n");
		printf(" 29 - Thinnes ratio\n");
		printf(" 30 - Extreme points of OBJ and aspect ratio\n");
		printf(" 31 - Projections of the binary objects\n");
		printf(" 32 - Object labels counter\n");
		printf(" 33 - Object labelling preview\n");
		printf(" 0  - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				negative_image();
				break;
			case 11:
				show_initial_image_grayscale("Images/kids.bmp");
				scanf("%d", &factor);
				additive_factor(factor);
				break;
			case 12:
				show_initial_image_grayscale("Images/kids.bmp");
				scanf("%d", &factor);
				multiplicative_factor(factor);
				break;
			case 13:
				create4squares_matrix();
				break;
			case 14:
				inverse();
				break;
			case 15:
				color_to_greyscale_components(imread("Images/kids.bmp", CV_LOAD_IMAGE_COLOR));
				break;
			case 16:
				color_to_greyscale(imread("Images/kids.bmp", CV_LOAD_IMAGE_COLOR));
				break;
			case 17:
				scanf("%d", &factor);
				grayscale_to_black_and_white(imread("Images/kids.bmp", CV_LOAD_IMAGE_GRAYSCALE),factor);
				break;
			case 18:
				rgb_to_hsv();
				break;
			case 19:
				char fname[MAX_PATH];
				scanf("%d", &x);
				scanf("%d", &y);
				while (openFileDlg(fname))
					isInside(imread(fname), x, y);
				break;
			case 20:
				histogram();
				break;
			case 21:
				yellow_detection();
				break;
			case 22:
				pdf();
				break;
			case 23:
				a1 = multilevel_thresholding(pdf());
				print_threshold_image(a1);
				waitKey(0);
				break;
			case 25:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				for(int i = 0; i < 100; i++)
					printf("%d\n", area(img, 0));
				waitKey(0);
				waitKey();
				break;
			case 26:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				printf("%d %d\n", get_I_COM(img, 0), get_J_COM(img, 0));
				for (int k = 0; k < 8; k++) {
					img.at<uchar>(get_I_COM(img, 0) + a[k], get_J_COM(img, 0) + b[k]) = 255;
				}
				img.at<uchar>(get_I_COM(img, 0), get_J_COM(img, 0)) = 255;
				imshow("Center of mass", img);
				waitKey(0);
				break;
			case 27:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				pStart.y = get_I_COM(img, 0);
				pStart.x = get_J_COM(img, 0);
				pEnd.y = pStart.y + sin(axis_of_elongation(img, 0))*length;
				pEnd.x = pStart.x + cos(axis_of_elongation(img, 0))*length;
				line(img, pStart, pEnd, 127);
				printf("%lf\n", axis_of_elongation(img, 0));
				imshow("axis of elongation", img);
				waitKey(0);
				break;
			case 28:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				printf("Perimeter = %d", perimeter(img));
				imshow("Initial image", img);
				waitKey(0);
				break;
			case 29:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				printf("%lf\n", thiness_ratio(img, 0));
				imshow("init", img);
				waitKey(0);
				break;
			case 30:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			
				printf("%d %d %d %d\n", row_min(img), col_min(img), row_max(img), col_max(img));
				printf("%lf\n", (float)(col_max(img) - col_min(img) + 1) / (float)(row_max(img) - row_min(img) + 1));
				
				line(img, Point(col_min(img), row_min(img)), Point(col_min(img), row_max(img)), 127, 2);
				line(img, Point(col_min(img), row_min(img)), Point(col_max(img), row_min(img)), 127, 2);
				line(img, Point(col_max(img), row_min(img)), Point(col_max(img), row_max(img)), 127, 2);
				line(img, Point(col_min(img), row_max(img)), Point(col_max(img), row_max(img)), 127, 2);
				imshow("Circumscribed rectangle", img);
				waitKey(0);
				break;
			case 31: 
				projection_horizontal(img1);
				printf("aaa");
				projection_vertical(img1);
				printf("aaa");
				for (int i = 0; i < img1.rows; i++) {
					for (int j = 0; j < img1.cols; j++) {
						if (horiz[i] <= j)
							buff1.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
						else
							buff1.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
					}
				}
				for (int i = 0; i < img1.cols; i++) {
					for (int j = 0; j < img1.rows; j++) {
						if (vert[i] <= j)
							buff2.at<Vec3b>(j, i) = Vec3b(255, 0, 0);
						else
							buff2.at<Vec3b>(j, i) = Vec3b(0, 255, 0);
					}
				}
				imshow("Initial", img1);
				imshow("Horiz", buff1);
				imshow("Vert", buff2);
				waitKey(0);
				break;
			case 32:
				img = imread("Images/letters.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				labelled = breadth_first_labelling(img);
				imshow("Black and white", img);
				imshow("Labelled image", labelled);
				waitKey(0);
				break;
			case 33:
				img = imread("Images/letters.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				buff = imread("Images/letters.bmp", CV_LOAD_IMAGE_COLOR);
				labelled = breadth_first_labelling(img);
				memset(used, 0, 256*sizeof(uchar));
				for (int k = 1; k < 255; k++) {
					std::random_device rd; // obtain a random number from hardware
					std::mt19937 eng(rd()); // seed the generator
					std::uniform_int_distribution<> distr(0, 255); // define the range
					r = distr(eng);
					g = distr(eng);
					b1 = distr(eng);
					maybe = distr(eng);
					for (int i = 0; i < labelled.rows; i++) {
						for (int j = 0; j < labelled.cols; j++) {
							if (k == labelled.at<uchar>(i, j)) {
								labelled.at<uchar>(i, j) = (uchar)maybe;
								buff.at<Vec3b>(i, j) = Vec3b(r, g, b1);
							}
						}
					}

				}
				imshow("Black and white", img);
				imshow("Labelled image", buff);
				waitKey(0);
				break;
			case 24:
				img = imread("Images/Single Object/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				Mat buff(img.rows, img.cols, CV_8UC1);
				for (int i = 0; i < buff.rows; i++) {
					for (int j = 0; j < buff.cols; j++) {
						if (check_if_in_object(img, i, j, 0) == 1)
							buff.at<uchar>(i, j) = 127;
						else buff.at<uchar>(i, j) = 255;
					}
				}
				imshow("Initial image", img);
				imshow("Checked if inside", buff);
				waitKey(0);
				break;
			
			
		}
	}
	while (op!=0);
	
	return 0;
}