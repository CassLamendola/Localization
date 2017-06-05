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

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = 1000;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

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
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise
	normal_distribution<double> dist_x(x, std_pos[0]);
  normal_distribution<double> dist_y(y, std_pos[1]);
  normal_distribution<double> dist_theta(theta, std_pos[2]);

  for (int i; i<num_particles; i++){
  	if (fabs(yaw_rate) > 0.0001){
  		particles[i].x += velocity/yaw_rate*(sin(particles[i].theta+(yaw_rate*delta_t))-sin(particles[i].theta));
  		particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+(yaw_rate*delta_t)));
  		particles[i].theta += (yaw_rate*delta_t);
  	}
  	else {
  		particles[i].x += velocity*delta_t*cos(particles[i].theta);
  		particles[i].y += velocity*delta_t*sin(particles[i].theta);
  	}

  	particles[i].x += dist_x(gen);
  	particles[i].y += dist_y(gen);
  	particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement
  for (int i=0; i<observations.size(); i++){
  	LandmarkObs observation = observations[i];

  	// Keep track of id for closest measurements
  	int landmark_id;

  	// Set minimum distance to the distance between first opbservation and first particle
  	float min_distance = dist(observation.x, observation.y, predicted[i].x, predicted[i].y)
  	
  	for(int j=0; j<particles.size(); j++){
  		LandmarkObs predicted_measurement = predicted[j];
  		distance = dist(observation.x, observation.y, predicted_measurement.x, predicted_measurement.y);
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
  // Update the weights of each particle using a mult-variate Gaussian distribution
  

  // You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems. Keep in mind that this transformation requires
  //   both rotation AND translation (but no scaling). The following is a good
  //   resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight. NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
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
