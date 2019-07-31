#include <random>
#include <iostream>
#include <time.h>
#include <set>
#include <cassert>
#include <limits.h>


using namespace std;
//数据点类型
struct Point2D {
	Point2D() {};
	Point2D(double X, double Y) :x(X), y(Y) {};
	double x;
	double y;
};
/**
  * @brief 线性模型
  *
  * Ax+By+C = 0;
*/
class LineModel {

	//待估计参数
	double A_, B_, C_;
public:
	LineModel() {};
	~LineModel() {};

	//使用两个点对直线进行初始估计
	void FindLineModel(const Point2D& pts1, const Point2D& pts2) {

		A_ = pts2.y - pts1.y;
		B_ = pts1.x - pts2.x;
		C_ = (pts1.y - pts2.y) * pts2.x + (pts2.x - pts1.x) * pts2.y;
	}

	//返回点到直线的距离
	double computeDistance(const Point2D& pt) {
		return abs(A_ * pt.x + B_ * pt.y + C_) / sqrt(A_ * A_ + B_ * B_);
	}

	//模型参数输出
	void printLineParam()
	{
		cout << "best_model.A =  "<< A_<< endl
			<<"  best_model.B =  " <<B_ <<endl
			<< " best_model.C=  "<<C_<< endl;
	}

	//利用最大的内点集重新估计模型
	//y=kx+b
	//利用最小二乘法：
	//k=(meanXY-meanX*meanY)/(meanXX-meanX*meanX)
	//b=meanY-k*meanX
	double estimateModel(vector<Point2D>& data, set<size_t>& inliers_set)
	{
		assert(inliers_set.size() >= 2);
		//求均值 means
		double meanX=0,meanY=0;
		double meanXY = 0, meanXX = 0;
		for (auto& idx : inliers_set) {
			meanX += data[idx].x;
			meanY += data[idx].y;
			meanXY += data[idx].x * data[idx].y;
			meanXX += data[idx].x * data[idx].x;

		}
		meanX /= inliers_set.size();
		meanY /= inliers_set.size();
		meanXY /= inliers_set.size();
		meanXX /= inliers_set.size();


		bool isVertical = (meanXX-meanX * meanX) == 0;
		double k = NAN;
		double b = NAN;

		if (isVertical)
		{
			A_ = 1;
			B_ = 0;
			C_ = meanX;
		}
		else
		{
			k = (meanXY - meanX * meanY) / (meanXX - meanX * meanX); 
			b = meanY - k * meanX;                                                 
			//A^2+B^2 = 1;
			//这里要注意k的符号
			double scaleFactor = (k>=0.0?1.0:-1.0) / sqrt(1 + k * k);
			A_ = scaleFactor * k;
			B_ = -scaleFactor;
			C_ = scaleFactor * b;
		}

		//误差计算
		double sXX, sYY, sXY;
		sXX = sYY = sXY = 0;
		for (auto& index : inliers_set) {
			Point2D point;
			point = data[index];
			sXX += (point.x - meanX) * (point.x - meanX);
			sYY += (point.y - meanY) * (point.y - meanY);
			sXY += (point.x - meanX) * (point.y - meanY);
		}
		double error = A_ * A_ * sXX + 2 * A_ * B_ * sXY + B_ * B_ * sYY;
		error /= inliers_set.size();
		return error;
	}

};



/**
* @brief 运行RANSAC算法
*
* @param[in]    data               一组观测数据
* @param[in]    n	           适用于模型的最少数据个数
* @param[in]    maxIterations       算法的迭代次数
* @param[in]    d                   判定模型是否适用于数据集的数据数目,于求解出来的模型的内点质量或者说数据集大小的一个约束
* @param[in]    t					用于决定数据是否适应于模型的阀值
* @param[in&out]    model           自定义的待估计模型，为该函数提供Update、computeError和Estimate三个成员函数
*                                    运行结束后，模型参数被设置为最佳的估计值
* @param[out]    best_consensus_set    输出一致点的索引值
* @param[out]    best_error         输出最小损失函数
*/

int runRansac(vector<Point2D>& dataSets, int n, int maxIterations, double sigma,int d,
		LineModel& best_model, set<size_t>& inlier_sets, double& best_error) {


		int isFound = 0;           //算法成功的标志
		int N = dataSets.size();
		set<int> maybe_inliers;    //初始随机选取的点（的索引值）
		LineModel currentModel;
		set<size_t> maxInliers_set;  //最大内点集
		best_error = 1.7976931348623158e+308;
		default_random_engine rng(time(NULL));     //随机数生成器
		uniform_int_distribution<int> dist(0, N - 1);  //采用均匀分布



	     //1. 新建一个容器allIndices，生成0到N-1的数作为点的索引
		vector<size_t> allIndices;
		allIndices.reserve(N);
		vector<size_t> availableIndices;

		for (int i = 0; i < N; i++)
		{
			allIndices.push_back(i);
		}


	     //2.这个点集是用来计算线性模型的所需的最小点集
		vector< vector<size_t> > minSets = vector< vector<size_t> >(maxIterations, vector<size_t>(n, 0));
		//随机选点，注意避免重复选取同一个点
		for (int it = 0; it < maxIterations; it++)
		{
			availableIndices = allIndices;
			for (size_t j = 0; j < n; j++)
			{
				// 产生0到N-1的随机数
				int randi = dist(rng);
				// idx表示哪一个索引对应的点被选中
				int idx = availableIndices[randi];

				minSets[it][j] = idx;
				//cout << "idx:" << idx << endl;
				// randi对应的索引已经被选过了，从容器中删除
				// randi对应的索引用最后一个元素替换，并删掉最后一个元素
				availableIndices[randi] = availableIndices.back();
				availableIndices.pop_back();
			}
		}


		//3.主循环程序，求解最大的一致点集
		vector<Point2D> pts;
		pts.reserve(n);
		for (int it = 0; it < maxIterations; it++)
		{

			for (size_t j = 0; j < n; j++)
			{
				int idx = minSets[it][j];

				pts[j] = dataSets[idx];

			}

			//cout << pts[0].x << endl << pts[1].x << endl;
			//根据随机到的两个点计算直线的模型
			currentModel.FindLineModel(pts[0],pts[1]);  

			//currentModel.printLineParam();
			set<size_t> consensus_set;        //选取模型后，根据误差阈值t选取的内点(的索引值)

			//根据初始模型和阈值t选择内点  
			// 基于卡方检验计算出的阈值
			const double th =sqrt(3.841*sigma*sigma);
			double current_distance_error = 0.0 ;
			double distance_error = 0.0;
			for (int i = 0; i < N; i++) 
			{
				current_distance_error= currentModel.computeDistance(dataSets[i]);
				if (current_distance_error < th) {
					consensus_set.insert(i);
				}

			}

			if (consensus_set.size() > maxInliers_set.size()) {
				maxInliers_set = consensus_set;
			}
		}

		//4.根据全部的内点重新计算模型
	        //重新在内点集上找一个误差最小的模型
		if (maxInliers_set.size() > d) {
			double current_distance_error = best_model.estimateModel(dataSets, maxInliers_set);
			//若当前模型更好，则更新输出量
			if (best_error < current_distance_error) {
				best_model = currentModel;
				inlier_sets = maxInliers_set;
				best_error = current_distance_error;
				isFound = 1;
			}
		}
		return isFound;
	}







int main() {

	vector<Point2D> Point_sets{ Point2D(3, 4), Point2D(6, 8), Point2D(9, 12), Point2D(15, 20), Point2D(10,-10) };
        //vector<Point2D> Point_sets{ Point2D(3, 4), Point2D(3, 7), Point2D(10,-10) };
        //vector<Point2D> Point_sets{ Point2D(3, 4), Point2D(6, 4), Point2D(9, 4), Point2D(15, 4), Point2D(10,-10) };
	int data_size = Point_sets.size();
	//2.设置输入量
	int maxIterations = 50;   //最大迭代次数
	int n = 2;                //模型自由度
	double t = 0.01;        //用于决定数据是否适应于模型的阀值
	int d = data_size * 0.5; //判定模型是否适用于数据集的数据数目
	//3.初始化输出量
	LineModel best_model;            //最佳线性模型
	set<size_t> best_consensus_set;  //记录一致点索引的set
	double best_error;              //最小残差
	//4.运行RANSAC            
	int status = runRansac(Point_sets, n, maxIterations, t, d, best_model, best_consensus_set, best_error);
	//5.输出
	best_model.printLineParam();
	return 0;
}
