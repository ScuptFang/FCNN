#pragma once
#include <algorithm>
#include <memory>
#include <complex>
#include <random>

class Preceptron {
public:
	Preceptron() {
	}
	Preceptron(int inputNum){
		Construct(inputNum);
	}
	~Preceptron() {

	}

	virtual void Construct(int inputNum) {
		srand(clock());
		weight.resize(inputNum);
		delta_weight.resize(inputNum);
		delta_bias.resize(inputNum);
		input.resize(inputNum);

		bias = (double)(rand() % 1000) / 1000;
		for (int i = 0; i < inputNum; i++) {
			weight[i] = (double)(rand() % 1000) / 1000;
			delta_weight[i] = 0.0;
			delta_bias[i] = 0.0;
		}

		motivation = 0.2;
		learning_step = 1.2;
		useBias = false;
	}

	virtual double Activiation(double X) = 0;

	virtual double Derivation(double X) = 0;

	virtual double Forward() = 0;

	virtual void Backward() = 0;

	virtual void Import() {
		//Import the network as you want.
	};

	virtual void Export() {
		//Export the network as you want.
	};

public:
	//forward parameters
	std::vector<double> weight;
	double bias;

	//backward parameters
	std::vector<double> delta_weight;
	std::vector<double> delta_bias;

	//general setting
	double motivation;
	double learning_step;
	bool useBias;

	//temporary data 
	std::vector<double> input;
	double output;
	double error;
};

class Preceptron_SIGMOID : public Preceptron {
public:
	double Activiation(double X) {
		return 1.0 / (1.0 + std::exp(-X));
	}

	double Derivation(double X) {
		double sig = Activiation(X);
		return sig * (1.0 - sig);
	}

	double Forward() {
		output = 0.0;
		for (int i = 0; i < weight.size(); i++) {
			output += useBias ? (weight[i] * input[i] + bias) : (weight[i] * input[i]);
		}
		output = Activiation(output);
		return output;
	}

	void Backward() {
		for (int i = 0; i < weight.size(); i++) {
			double derivation = Derivation(input[i]);
			double internal_weight_error = input[i] * derivation * error;
			delta_weight[i] = weight[i] * internal_weight_error;
			weight[i] += delta_weight[i] * motivation + learning_step * internal_weight_error;
			delta_bias[i] = derivation * error;
		}
		if (useBias) {
			double average_bias_error = 0.0;
			for (int i = 0; i < delta_bias.size(); i++) {
				average_bias_error += delta_bias[i];
			}average_bias_error /= delta_bias.size();
			bias += learning_step * average_bias_error;
		}
	}
};

