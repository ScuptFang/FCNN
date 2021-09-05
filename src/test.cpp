#include "perceptron.hpp"
#include "FC_BPNN.h"
#include <iostream>


double average4(double a, double b, double c, double d) {
	return (a + b + c + d) / 4;
}

int main_average() {
	FC_BPNN bp(4, 10, 1);
	std::vector<double> data; data.push_back(0.251); data.push_back(0.425); data.push_back(0.775); data.push_back(0.952);
	std::vector<double> output;
	double lastCost = 0.0; double cost = 0.0;
	for (int i = 1; i <= 100; i++) {
		double a = (double)(rand() % 1000) / 1000;
		double b = (double)(rand() % 1000) / 1000;
		double c = (double)(rand() % 1000) / 1000;
		double d = (double)(rand() % 1000) / 1000;
		cost += bp.train_once({ a ,b, c,d }, { average4(a,b,c,d) });
		if (i % 10 == 0) {
			cost /= 10;
			std::cout << "iter: " << i << " cost: " << cost << " step: " << bp.getLearningStep() << std::endl;
			lastCost = cost;
			cost = 0.0;
		}
	}
	bp.predict(data, output);
	std::cout << "prediction: " << output[0] << " gt: " << average4(data[0], data[1], data[2], data[3]) << std::endl;
	return 1;
}

double square_root_add_cube_root(double a, double b) {
	return sqrt(a) + cbrt(b);
}

int main_sqrtAddcbrt() {
	FC_BPNN bp(2, 6, 7, 1);
	std::vector<double> data; data.push_back(0.251); data.push_back(0.0274);
	std::vector<double> output;
	double lastCost = 0.0; double cost = 0.0;
	for (int i = 1; i <= 1000; i++) {
		double a = (double)(rand() % 1000) / 1000;
		double b = (double)(rand() % 1000) / 1000;
		cost += bp.train_once({ a ,b }, { square_root_add_cube_root(a,b) });
		if (i % 100 == 0) {
			cost /= 100;
			std::cout << "iter: " << i << " cost: " << cost << " step: " << bp.getLearningStep() << std::endl;
			double costRate = (lastCost - cost) / lastCost;
			if (costRate <= 0.1) {
				bp.setLearningStep(bp.getLearningStep() / 2);
			}
			lastCost = cost;
			cost = 0.0;
		}
	}
	bp.predict(data, output);
	std::cout << "prediction: " << output[0] << " gt: " << square_root_add_cube_root(data[0], data[1]) << std::endl;
	return 1;
}

int main_FCBPNN_classify() {
	FC_BPNN bp(2, 10, 1);

	std::vector<std::vector<double>> samples;
	std::vector<std::vector<double>> labels;
	samples.push_back({ 1.0,0.0 }); labels.push_back({ 1.0 });
	samples.push_back({ 0.0,1.0 }); labels.push_back({ 1.0 });
	samples.push_back({ 1.0,1.0 }); labels.push_back({ 0.0 });
	samples.push_back({ 0.0,0.0 }); labels.push_back({ 0.0 });
	double cost = 0.0;
	for (int i = 1; i <= 2000; i++) {
		cost += bp.train_once(samples[i % 4], labels[i % 4]);
		if (i % 100 == 0) {
			cost /= 100;
			std::cout << "iter: " << i << " cost: " << cost << " step: " << bp.getLearningStep() << std::endl;
			cost = 0.0;
		}
	}
	std::vector<double> output;
	bp.predict({ 0.1, 0.2}, output);
	std::cout << "prediction: " << output[0] << std::endl;
	bp.predict({ 0.21, 0.9 }, output);
	std::cout << "prediction: " << output[0] << std::endl;
	return 1;
}

int main() {
	main_average();
	main_sqrtAddcbrt();
	main_FCBPNN_classify();
	return 1;
}