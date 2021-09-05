#pragma once
#include "perceptron.hpp"
#include <cstdarg>
#include <vector>
#include <map>


class FC_BPNN {
public:
	FC_BPNN(int perceptronNum, ...);
	~FC_BPNN();

	double train_once(const std::vector<double>& sample, const std::vector<double>& label);
	void predict(const std::vector<double>& data, std::vector<double>& prediction);
	void setLearningStep(double a);
	double getLearningStep();

private:
	bool setSample(const std::vector<double>& sample);
	bool setLabel(const std::vector<double>& label);
	double getCost();

	bool Forward();
	bool Backward();

	void clear();

	std::vector<std::vector<double>> layers_value;
	std::vector<std::vector<double>> layers_error;
	std::vector<std::map<size_t, Preceptron*>> net;

};
