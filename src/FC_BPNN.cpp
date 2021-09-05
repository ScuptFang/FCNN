#include "FC_BPNN.h"
#include <iostream>

FC_BPNN::FC_BPNN(int perceptronNum, ...)
{
	va_list ap;
	va_start(ap, 0);
	int num = va_arg(ap, int);
	while (num > -1)
	{
		std::vector<double> layer_value(num, 0.0);
		std::vector<double> layer_loss(num, 0.0);
		layers_value.push_back(layer_value);
		layers_error.push_back(layer_loss);
		num = va_arg(ap, int);
	}
	va_end(ap);

	for (int i = 1; i < layers_value.size(); i++) {
		std::map<size_t, Preceptron*> subNet;
		for (int j = 0; j < layers_value[i - 1].size(); j++) {
			for (int k = 0; k < layers_value[i].size(); k++) {
				size_t finder0 = j;
				size_t finder1 = k;
				size_t finder = (finder0 << 32) | finder1;
				auto ptr = dynamic_cast<Preceptron*>(new Preceptron_SIGMOID());
				ptr->Construct(layers_value[i - 1].size());
				subNet[finder] = ptr;
			}
		}
		net.push_back(subNet);
	}

}

FC_BPNN::~FC_BPNN()
{
	for (int i = 0; i < net.size(); i++) {
		for (auto& element : net[i]) {
			delete element.second;
			element.second = nullptr;
		}
	}
}

double FC_BPNN::train_once(const std::vector<double>& sample, const std::vector<double>& label)
{
	clear();
	setSample(sample);
	Forward();
	setLabel(label);
	Backward();
	return getCost();
}

void FC_BPNN::predict(const std::vector<double>& data, std::vector<double>& prediction)
{
	clear();
	setSample(data);
	Forward();
	int lastIndex = layers_value.size() - 1;
	prediction = layers_value[lastIndex];
}

void FC_BPNN::setLearningStep(double a)
{
	for (int i = 0; i < net.size(); i++) {
		for (auto& element : net[i]) {
			element.second->learning_step = a;
		}
	}
}

double FC_BPNN::getLearningStep()
{
	for (int i = 0; i < net.size(); i++) {
		for (auto& element : net[i]) {
			return element.second->learning_step;
		}
	}
}

bool FC_BPNN::setSample(const std::vector<double>& sample)
{
	if (sample.size() != layers_value.front().size()) {
		std::cerr << "Error input!" << std::endl;
		return false;
	}
	for (int j = 0; j < layers_value[0].size(); j++) {
		layers_value[0][j] = sample[j];
	}
	return true;
}

bool FC_BPNN::setLabel(const std::vector<double>& label)
{
	if (label.size() != layers_value.back().size()) {
		std::cerr << "Error input!" << std::endl;
		return false;
	}
	for (int j = 0 ; j < layers_error.back().size(); j++) {
		layers_error.back()[j] = label[j] - layers_value.back()[j];
	}
	return true;
}

double FC_BPNN::getCost()
{
	double cost = 0.0;
	int lastIndex = layers_error.size() - 1;
	for (int j = 0; j < layers_error[lastIndex].size(); j++) {
		cost += layers_error[lastIndex][j] * layers_error[lastIndex][j];
	}
	cost /= layers_error[lastIndex].size();
	return sqrt(cost);
}

bool FC_BPNN::Forward()
{
	for (int i = 0; i < net.size(); i++) {
		for (auto& element : net[i]) {
			for (int j = 0; j < layers_value[i].size(); j++)
				element.second->input[j] = layers_value[i][j];
			size_t finder = element.first;
			size_t finder1 = finder & 0X00000000FFFFFFFF;
			dynamic_cast<Preceptron_SIGMOID*>(element.second)->Forward();
			layers_value[i + 1][finder1] += element.second->output;
		}
	}
	return true;
}

bool FC_BPNN::Backward()
{
	for (int i = net.size() - 1; i >= 0; i--) {
		for (auto& element : net[i]) {
			size_t finder = element.first;
			size_t finder1 = finder & 0X00000000FFFFFFFF;
			element.second->error = layers_error[i + 1][finder1];
			dynamic_cast<Preceptron_SIGMOID*>(element.second)->Backward();
			for (int j = 0; j < element.second->delta_weight.size(); j++) {
				layers_error[i][j] += element.second->delta_weight[j];
			}
		}
	}
	return true;
}

void FC_BPNN::clear()
{
	for (int i = 0; i < layers_value.size(); i++) {
		for (int j = 0; j < layers_value[i].size(); j++) {
			layers_value[i][j] = 0.0;
			layers_error[i][j] = 0.0;
		}
	}
}
