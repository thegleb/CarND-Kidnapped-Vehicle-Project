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
  
  // this worked fine with 10 particles
  // 100 worked as well
  // 50 is a happy medium for performance and accuracy
  num_particles = 50;
  
  std::random_device rd;
  
  std::default_random_engine x_generator(rd());
  std::default_random_engine y_generator(rd());
  std::default_random_engine theta_generator(rd());
  
  std::normal_distribution<double> x_dist(0, std[0]);
  std::normal_distribution<double> y_dist(0, std[1]);
  std::normal_distribution<double> theta_dist(0, std[2]);
  
  // initialize the particles with locations around the initial measurement with random noise added
  // based on the standard deviations given
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    
    particle.id = i;
    particle.x = x + x_dist(x_generator);
    particle.y = y + y_dist(y_generator);
    particle.theta = theta + theta_dist(theta_generator);
    particle.weight = 1;
    
    particles.push_back(particle);
  }
  
  // this lets the rest of the algorithm proceed
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */

  std::random_device rd;

  // separate generators for x, y, and theta for extra randomness
  std::default_random_engine x_generator(rd());
  std::default_random_engine y_generator(rd());
  std::default_random_engine theta_generator(rd());
  
  std::normal_distribution<double> x_dist(0, std_pos[0]);
  std::normal_distribution<double> y_dist(0, std_pos[1]);
  std::normal_distribution<double> theta_dist(0, std_pos[2]);
  
  for (int i = 0; i < particles.size(); i++) {
    // if the yaw_rate approaches 0, we start to get invalid values (divide by 0)
    // so when the yaw rate is small, we switch to a regular linear prediction for x/y
    if (abs(yaw_rate) > MIN_YAW_RATE) {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + x_dist(x_generator);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + y_dist(y_generator);
    } else {
      particles[i].x += velocity * delta_t * sin(particles[i].theta) + x_dist(x_generator);
      particles[i].y += velocity * delta_t * cos(particles[i].theta) + y_dist(y_generator);
    }

    // using yaw_rate here shouldn't hurt even when it is small; this should improve predictions
    particles[i].theta += yaw_rate * delta_t + theta_dist(theta_generator);
  }
  
}

struct Association {
  int id;
  double distance;
};

void ParticleFilter::dataAssociation(vector<Map::single_landmark_s> landmarks,
                                     vector<LandmarkObs>& observations, Particle& particle) {
  /**
   * Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   */
  Map::single_landmark_s nearest;
  
  // debug info for each particle
  vector<int> associations;
  vector<double> sense_x;
  vector<double> sense_y;

  for (int i = 0; i < observations.size(); ++i) {

    // start with infinity as the closest landmark for each observation
    double minDistance = INFINITY;

    for (int j = 0; j < landmarks.size(); ++j) {
      double currDistance = dist(landmarks[j].x_f, landmarks[j].y_f, observations[i].x, observations[i].y);

      // if we found a closer landmark, then update nearest/minDistance
      if (currDistance < minDistance) {
        nearest = landmarks[j];
        minDistance = currDistance;
      }
    }

    // set the nearest landmark id on the observation
    observations[i].id = nearest.id_i;

    // update debug info for each particle
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
   * Update the weights of each particle using a multi-variate Gaussian distribution.
   */

  vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
  
  double sum_weights = 0;
  for (int i = 0; i < particles.size(); i++) {
    vector<Map::single_landmark_s> landmarks_in_range;

    // 1. find all the landmarks in range of the particle to avoid false matches
    for (int j = 0; j < landmarks.size(); j++) {
      if (dist(particles[i].x, particles[i].y, landmarks[j].x_f, landmarks[j].y_f) <= sensor_range) {
        landmarks_in_range.push_back(landmarks[j]);
      }
    }

    // 2. transform current observations to map coordinates
    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs transformed = transform_to_map_coord(particles[i].x, particles[i].y, particles[i].theta,
                                                       observations[j].x, observations[j].y);
      // also copy over the id
      transformed.id = observations[j].id;

      transformed_observations.push_back(transformed);
    }

    // 3. associate transformed observations with predicted landmarks
    dataAssociation(landmarks_in_range, transformed_observations, particles[i]);

    // 4. calculate probability using multi the variate gaussian distribution
    double prob = 1.0f;
    for (int j = 0; j < transformed_observations.size(); j++) {
      int landmark_id = transformed_observations[j].id - 1;

      // disregard invalid IDs
      // (probably not needed after the rest of the code started working but keeping it just in case)
      if (landmark_id >= 0 and landmark_id <= landmarks.size()) {
        Map::single_landmark_s nearest_landmark = landmarks[transformed_observations[j].id - 1];
        double gaussian = multi_variate_gaussian(
                                                 transformed_observations[j].x,
                                                 transformed_observations[j].y,
                                                 nearest_landmark.x_f, nearest_landmark.y_f,
                                                 std_landmark[0], std_landmark[1]);
        prob *= gaussian;
      }
    }
    
    // 5. update particle weight using the product of the probabilities of the observations
    //    e.g. what is the overall probability of this location being correct considering
    //    these exact observations
    particles[i].weight = prob;
    
    // 6. accumulate the sum of all weights (we're iterating over each particle already)
    sum_weights += prob;
  }
 
  // 7. normalize weights using the sum accumulated in step 6
  for (int i = 0; i < particles.size(); i++) {
    if (sum_weights > 0) {
      particles[i].weight = particles[i].weight / sum_weights;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional to their weight.
   */

  vector<Particle> new_particles;
  vector<double> probabilities;

  // extract previously calculated probabilities (normalized particle weights)
  // and push them to a vector
  for (int i = 0; i < particles.size(); ++i) {
    probabilities.push_back(particles[i].weight);
  }

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution (probabilities.begin(), probabilities.end());

  // discrete_distribution selects items with replacement using the probabilities we calculated
  // we need to repeat this process N times for N particles
  for (int i = 0; i < particles.size(); ++i) {
    int selected_idx = distribution(generator);
    new_particles.push_back(particles[selected_idx]);
  }

  // update the particle filter particles with the resampled particle vector
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
