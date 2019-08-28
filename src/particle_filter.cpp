/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles and initializes all particles to the first position
   * (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
   * Adds random Gaussian noise to each particle.
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::random_device rd;
  
  std::default_random_engine x_generator(rd());
  std::default_random_engine y_generator(rd());
  std::default_random_engine theta_generator(rd());
  
  std::normal_distribution<double> x_dist(0, std[0]);
  std::normal_distribution<double> y_dist(0, std[1]);
  std::normal_distribution<double> theta_dist(0, std[2]);
  
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    
    particle.id = i;
    particle.x = x + x_dist(x_generator);
    particle.y = y + y_dist(y_generator);
    particle.theta = theta + theta_dist(theta_generator);
    particle.weight = 1;
    
    particles.push_back(particle);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::random_device rd;
  
  std::default_random_engine x_generator(rd());
  std::default_random_engine y_generator(rd());
  std::default_random_engine theta_generator(rd());
  
  std::normal_distribution<double> x_dist(0, std_pos[0]);
  std::normal_distribution<double> y_dist(0, std_pos[1]);
  std::normal_distribution<double> theta_dist(0, std_pos[2]);
  
  for (int i = 0; i < particles.size(); i++) {
//    if (abs(yaw_rate) > MIN_YAW_RATE) {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + x_dist(x_generator);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + y_dist(y_generator);
      particles[i].theta += yaw_rate * delta_t + theta_dist(theta_generator);
//    } else {
//      particles[i].x += velocity * delta_t * sin(particles[i].theta) + x_dist(x_generator);
//      particles[i].y += velocity * delta_t * cos(particles[i].theta) + y_dist(y_generator);
//    }
  }
  
}

struct Association {
  int id;
  double distance;
};

void ParticleFilter::dataAssociation(vector<Map::single_landmark_s> landmarks,
                                     vector<LandmarkObs>& observations, Particle& particle) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  Map::single_landmark_s nearest;
//  vector<int, double> association;
  
  vector<int> associations;
  vector<double> sense_x;
  vector<double> sense_y;
  for (int i = 0; i < observations.size(); ++i) {
    double minDistance = INFINITY;
    for (int j = 0; j < landmarks.size(); ++j) {
      double currDistance = dist(landmarks[j].x_f, landmarks[j].y_f, observations[i].x, observations[i].y);
      if (currDistance < minDistance) {
        nearest = landmarks[j];
        minDistance = currDistance;
      }
    }

    observations[i].id = nearest.id_i;

    associations.push_back(nearest.id_i);
    sense_x.push_back(nearest.x_f);
    sense_y.push_back(nearest.y_f);
    SetAssociations(particle, associations, sense_x, sense_y);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
  
  double sum_weights = 0;
  for (int i = 0; i < particles.size(); i++) {
    
    vector<Map::single_landmark_s> landmarks_in_range;
    for (int i = 0; i < landmarks.size(); i++) {
      Map::single_landmark_s landmark = landmarks[i];
      if (dist(landmark.x_f, landmark.y_f, particles[i].x, particles[i].y) <= sensor_range) {
        landmarks_in_range.push_back(landmark);
      }
    }

    double prob = 1.0f;
    vector<LandmarkObs> transformed_observations;
    // transform observations to map coordinates
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs transformed = transform_to_map_coord(particles[i].x, particles[i].y, particles[i].theta, observations[j].x, observations[j].y);
      transformed.id = observations[j].id;
      transformed_observations.push_back(transformed);
    }
    // associate transformed observations with predicted landmarks
    dataAssociation(landmarks_in_range, transformed_observations, particles[i]);

    for (int j = 0; j < transformed_observations.size(); j++) {
      int landmark_id = transformed_observations[j].id - 1;
      // disregard invalid IDs
      if (landmark_id >= 0 and landmark_id <= landmarks.size()) {
        Map::single_landmark_s nearest_landmark = landmarks[transformed_observations[j].id - 1];
        double gaussian = multi_variate_gaussian(transformed_observations[j].x, transformed_observations[j].y, nearest_landmark.x_f, nearest_landmark.y_f, std_landmark[0], std_landmark[1]);
        prob *= gaussian;
      }
    }
    particles[i].weight = prob;
    sum_weights += prob;
  }
 
  // normalize weight
  for (int i = 0; i < particles.size(); i++) {
    if (sum_weights > 0) {
      particles[i].weight = particles[i].weight / sum_weights;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  vector<double> probabilities;

  for (int i = 0; i < particles.size(); ++i) {
    probabilities.push_back(particles[i].weight);
  }

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution (probabilities.begin(), probabilities.end());

  for (int i = 0; i < particles.size(); ++i) {
    int selected_idx = distribution(generator);
    new_particles.push_back(particles[selected_idx]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;
  
  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }
  
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
