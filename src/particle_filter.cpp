/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = 100;
  
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  // Initialize all particles to first position and set all weights to 1
  for (int i=0; i<num_particles; i++) {
  	Particle particle;
  	particle.id = i;
    particle.x = x;
    particle.y = y;
    particle.theta = theta;
  	particle.weight = 1.0;

  	// Add random Gaussian noise to each particle
  	particle.x += dist_x(gen);
  	particle.y += dist_y(gen);
  	particle.theta += dist_theta(gen);

  	particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  
  //  Add random Gaussian noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Add measurements to each particle
	for (int i=0; i<num_particles; i++){
		double new_x, new_y, new_theta;

  	if (yaw_rate == 0){
  		new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
  		new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
  		new_theta = particles[i].theta;
  	}
  	else {
  		new_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
  		new_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
  		new_theta = particles[i].theta + yaw_rate*delta_t;
  	}

  	particles[i].x = new_x + dist_x(gen);
  	particles[i].y = new_y + dist_y(gen);
  	particles[i].theta = new_theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement
  for (int i=0; i<observations.size(); i++){
  	LandmarkObs observation = observations[i];

  	// Keep track of id for closest measurements
  	int landmark_id = -1;

  	// Set minimum distance to maximum possible distance
  	double min_distance = numeric_limits<double>::max();
  	
  	for(int j=0; j<predicted.size(); j++){
  		LandmarkObs predicted_measurement = predicted[j];
  		double distance = dist(observation.x, observation.y, predicted_measurement.x, predicted_measurement.y);
  		if (distance < min_distance){
  			min_distance = distance;
  			landmark_id = predicted_measurement.id;
  		}
  	}
  	// Assign the observed measurement to this particular landmark
  	observations[i].id = landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // Get coordinates of each particle
  for (int i=0; i<num_particles; i++){
  	double x = particles[i].x;
  	double y = particles[i].y;
  	double theta = particles[i].theta;

    // Store locations of predicted landmarks that are inside the sensor range of the particle
    vector<LandmarkObs> predicted_landmarks;

    // Get coordinates for each landmark
    for (int k=0; k<map_landmarks.landmark_list.size(); k++){
      
      int l_id = map_landmarks.landmark_list[k].id_i;
      double l_x = map_landmarks.landmark_list[k].x_f;
      double l_y = map_landmarks.landmark_list[k].y_f;
      LandmarkObs current_landmark = {l_id, l_x, l_y};

      // Choose landmarks within sensor range of particle
      if (fabs(dist(l_x, l_y, x, y)) <= sensor_range){
        predicted_landmarks.push_back(current_landmark);
      }
    }
  	// Store transformed points
    vector<LandmarkObs> transformed_observations;

    // Transform points from vehicle's coordinate system to map's coordinate system
  	for (int j=0; j<observations.size(); j++){
  		double t_x = observations[j].x*cos(theta) - observations[j].y*sin(theta) + x;
  		double t_y = observations[j].x*sin(theta) + observations[j].y*cos(theta) + y;
  		
  		// Store id and coordinates for transformed point
  		LandmarkObs transformed_observation;
  		transformed_observation.id = observations[j].id;
  		transformed_observation.x = t_x;
  		transformed_observation.y = t_y;

  		transformed_observations.push_back(transformed_observation);
  	}
  	// Data Associations
  	dataAssociation(predicted_landmarks, transformed_observations);

  	double total_weight = 1.0;
    weights[i] = 1.0;
    vector<int> associations_vec;
    vector<double> sense_x_vec;
    vector<double> sense_y_vec;

  	// Get coordinates of each transformed observation
  	for (int l=0; l<transformed_observations.size(); l++){
  		int o_id = transformed_observations[l].id;
  		double o_x = transformed_observations[l].x;
  		double o_y = transformed_observations[l].y;
  		double p_x;
  		double p_y;

  		// Get coordinates of the predicted landmark for the transformed observation
  		for (int m=0; m<predicted_landmarks.size(); m++){
  			if (predicted_landmarks[m].id == o_id){
  				p_x = predicted_landmarks[m].x;
  				p_y = predicted_landmarks[m].y;
  			}
  		}
  		// Calculate each weight using a mult-variate Gaussian distribution
  		double w = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) *
  							exp(-(pow(p_x-o_x, 2)/(2*pow(std_landmark[0],2)) +
  								pow(p_y-o_y, 2)/(2*pow(std_landmark[1],2)) -
  								(2*(p_x-o_x)*(p_y-o_y)/(sqrt(std_landmark[0])*sqrt(std_landmark[1])))));

      total_weight *= w;
      associations_vec.push_back(o_id);
      sense_x_vec.push_back(o_x);
      sense_y_vec.push_back(o_y); 
  	}
    particles[i].weight = total_weight;
    weights[i] = total_weight;

    SetAssociations(particles[i], associations_vec, sense_x_vec, sense_y_vec);
    predicted_landmarks.clear();
  }

  // More about this distribution here (Bivariate case):
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
}

void ParticleFilter::resample() {
	// Create discrete distribution for weights
  discrete_distribution<int> index(weights.begin(), weights.end());

  // Store resampled particles
  vector<Particle> resampled_particles;

  // Resample particles with replacement with probability proportional to their weight
  for (int i=0; i<num_particles; i++){
  	resampled_particles.push_back(particles[index(gen)]);
  }
  // Save resampled particles
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
