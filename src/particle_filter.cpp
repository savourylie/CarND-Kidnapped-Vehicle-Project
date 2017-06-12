/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Initialize number of particles
	num_particles = 10000;

	// Create Gaussian generators
	std::default_random_engine rand_generator;
	std::normal_distribution<double> x_distro(0.0, std[0]);
	std::normal_distribution<double> y_distro(0.0, std[1]);
	std::normal_distribution<double> theta_distro(0.0, std[2]);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle temp_particle;

		double x_noise = x_distro(rand_generator);
		double y_noise = y_distro(rand_generator);
		double theta_noise = theta_distro(rand_generator);

		// Initialize all positions based on GPS and add noise
		temp_particle.x = x + x_noise;
		temp_particle.y = y + y_noise;
		temp_particle.theta = theta + theta_noise;

		// Initialize all weights to 1
		temp_particle.weight = 1;

		// Push to vector particles
		particles.push_back(temp_particle);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Create Gaussian generators
	std::default_random_engine rand_generator;
	std::normal_distribution<double> x_distro(0.0, std_pos[0]);
	std::normal_distribution<double> y_distro(0.0, std_pos[1]);
	std::normal_distribution<double> theta_distro(0.0, std_pos[2]);

	for (unsigned int i = 0; i < particles.size(); ++i) {
		double x_noise = x_distro(rand_generator);
		double y_noise = y_distro(rand_generator);
		double theta_noise = theta_distro(rand_generator);

		double temp_x = particles[i].x;
		double temp_y = particles[i].y;
		double temp_theta = particles[i].theta;

		particles[i].x = temp_x + (velocity / yaw_rate) * (sin(temp_theta + yaw_rate * delta_t) - sin(temp_theta)) + x_noise;
		particles[i].y = temp_y + (velocity / yaw_rate) * (cos(temp_theta) - cos(temp_theta + yaw_rate * delta_t)) + y_noise;
		particles[i].theta = temp_theta + yaw_rate * delta_t + theta_noise;		
	}
}

std::vector<int> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	std::vector<int> closest;

	for (unsigned int i = 0; i < observations.size(); ++i) {
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		double min = std::numeric_limits<double>::infinity();
		double min_idx;

		for (unsigned int j = 0; j < predicted.size(); ++j) {
			double predicted_x = predicted[j].x;
			double predicted_y = predicted[j].y;

			double dist = sqrt((obs_x - predicted_x)*(obs_x - predicted_x) + (obs_y - predicted_y)*(obs_y - predicted_y));
			
			if (dist < min) {
				min = dist;
				min_idx = j;
			}	
		}

		closest.push_back(min_idx);
	}

	return closest;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Create Gaussian generators
	double lm_std_x = std_landmark[0];
	double lm_std_y = std_landmark[1];

	std::default_random_engine rand_generator;
	std::normal_distribution<double> x_distro(0.0, lm_std_x);
	std::normal_distribution<double> y_distro(0.0, lm_std_y);

	// Loop thru all particles
	for (unsigned int i = 0; i < particles.size(); ++i) {
		// Get current particle coordinates and heading
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// Create rotation matrix
		double rot_mat[2][2];
		rot_mat[0][0] = cos(p_theta);
		rot_mat[0][1] = sin(-p_theta);
		rot_mat[1][0] = sin(p_theta);
		rot_mat[1][1] = cos(p_theta);
		
		// Declare landmarks_in_range and predicted landmarks
		std::vector<LandmarkObs> observed_lms;
		std::vector<LandmarkObs> predicted_lms;

		// Get landmarks_in_range (predicted landmarks)
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;

			double dist_predict = sqrt((p_x - landmark_x)*(p_x - landmark_x) + (p_y - landmark_y)*(p_y - landmark_y));
			
			if (dist_predict <= 50) {
				LandmarkObs predicted;
				predicted.x = landmark_x;
				predicted.y = landmark_y;

				predicted_lms.push_back(predicted);
			}
		}

		// Get observed landmarks
		for (unsigned int i = 0; i < observations.size(); ++i) {
			int obs_id = observations[i].id;
			double obs_x = observations[i].x + x_distro(rand_generator);
			double obs_y = observations[i].y + y_distro(rand_generator);;

			double trans_x = rot_mat[0][0] * obs_x + rot_mat[0][1] * obs_y; 
			double trans_y = rot_mat[1][0] * obs_x + rot_mat[1][1] * obs_y;

			double lm_x = trans_x + p_x;
			double lm_y = trans_y + p_y;

			double dist_obs = sqrt((p_x - lm_x)*(p_x - lm_x) + (p_y - lm_y)*(p_y - lm_y));

			if (dist_obs <= 50) {
				LandmarkObs obs;
				obs.x = lm_x;
				obs.y = lm_y;
				observed_lms.push_back(obs);
			}			
		}

		std::vector<int> closest_ids = dataAssociation(predicted_lms, observed_lms);

		double prob = 1.0;

		for (unsigned int k = 0; k < closest_ids.size(); ++k) {
			int lm_id = closest_ids[k];
			double mu_x = predicted_lms[lm_id].x;
			double mu_y = predicted_lms[lm_id].y;
			
			prob *= Gaussian2D(observed_lms[k].x, observed_lms[k].y, mu_x, mu_y, lm_std_x, lm_std_y);
		}

		weights.push_back(prob);
	}
}

double ParticleFilter::Gaussian2D(double x, double y, double mu_x, double mu_y, double std_x, double std_y) {
	double scaling_constant = 1 / (2 * M_PI * std_x * std_y);
	double index_x = (x - mu_x)*(x - mu_x) / (2 * std_x*std_x);
	double index_y = (y - mu_y)*(y - mu_y) / (2 * std_y*std_y);

	return scaling_constant * exp(-1 * (index_x + index_y));
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> particles_temp;

	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::map<int, int> m;

    for(unsigned int i = 0; i < num_particles; ++i) {
        particles_temp.push_back(particles[d(gen)]);
    }

	particles = particles_temp;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
