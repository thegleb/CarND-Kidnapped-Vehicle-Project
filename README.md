# Udacity SDC Engineer: Kidnapped Vehicle Project

## Project Introduction (shamelessly modified from the project readme)
 Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

This project implement a 2 dimensional particle filter in C++. The particle filter is given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter also receives observation and control data.

## Running the Code (also shamelessly preserved from the project readme)
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./particle_filter

Alternatively some scripts have been included to streamline this process, these can be leveraged by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh

INPUT: values provided by the simulator to the c++ program

// sense noisy position data from the simulator

["sense_x"]

["sense_y"]

["sense_theta"]

// get the previous velocity and yaw rate to predict the particle's transitioned state

["previous_velocity"]

["previous_yawrate"]

// receive noisy observation data from the simulator, in a respective list of x/y values

["sense_observations_x"]

["sense_observations_y"]


OUTPUT: values provided by the c++ program to the simulator

// best particle values used for calculating the error evaluation

["best_particle_x"]

["best_particle_y"]

["best_particle_theta"]

//Optional message data used for debugging particle's sensing and associations

// for respective (x,y) sensed positions ID label

["best_particle_associations"]

// for respective (x,y) sensed positions

["best_particle_sense_x"] <= list of sensed x positions

["best_particle_sense_y"] <= list of sensed y positions


## Implementing the Particle Filter

`src/main.cpp` contains the logic for communicating with the simulator code. It also defines the proper sequence of particle filter steps: initialize (if first step), predict, sense, update weights, resample, and finally output average and best weights.

The actual logic for each of the steps mentioned above (other than check accuracy) is implemented inside  `src/particle_filter.cpp` and described at a high level below.

### Initialization

Our `ParticleFilter` is provided an initial reading of a potential location. We start by initializing 50 particles (number arbitrarily chosen because it provides good accuracy while making the code run reasonably quickly).

```
for (int i = 0; i < num_particles; i++) {
  particle.x = x + x_dist(x_generator);
  particle.y = y + y_dist(y_generator);
  particle.theta = theta + theta_dist(theta_generator);
}
```

We add random noise to the x/y/theta readings to provide a variety of perspectives - after all, the particle filter logic will decide which particle is most likely to represent the correct position of the robot.

### Prediction

If the particle filter has already been initialized, then we use current `velocity`, `yaw_rate` (how fast the direction is changing), as well as the time step (`delta_t`) to predict a new position and direction for each particle. To each of the predictions we also add a bit of random Gaussian noise.

```
particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + x_dist(x_generator);
particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + y_dist(y_generator);
particles[i].theta += yaw_rate * delta_t + theta_dist(theta_generator);
```

When yaw rate is close to 0, the formulas above become problematic as we might be dividing by zero when `yaw_rate` is 0; for sufficiently small `yaw_rate`s we do a simple linear prediction for the new position:

```
particles[i].x += velocity * delta_t * sin(particles[i].theta) + x_dist(x_generator);
particles[i].y += velocity * delta_t * cos(particles[i].theta) + y_dist(y_generator);
```

### Sensing

After the new prediction is calculated, we first have to transform each of the observations at that location from the robot coordinate system into map space with a little trigonometry, where `x_pos` and `y_pos` represents the x/y coordinates of each particle and `x_obs` / `y_obs` represent the x/y offsets of each observation from the robot's perspective.

```
transformed.x = x_pos + cos(heading) * x_obs - sin(heading) * y_obs;
transformed.y = y_pos + sin(heading) * x_obs + cos(heading) * y_obs;
```

The next step is to filter out any landmark that is too far from the current particle location to make associating observations with the landmarks more accurate.

```
if (dist(particles[i].x, particles[i].y, landmarks[j].x_f, landmarks[j].y_f) <= sensor_range) {
  landmarks_in_range.push_back(landmarks[j]);
}
```

Now we can associate each observation with the corresponding landmarks by doing a  simple nearest neighbors search. In condensed form, it looks like this:

```
double minDistance = INFINITY;
for (int j = 0; j < landmarks.size(); ++j) {
  double currDistance = dist(landmarks[j].x_f, landmarks[j].y_f, observations[i].x, observations[i].y);

  if (currDistance < minDistance) {
    nearest = landmarks[j];
    minDistance = currDistance;
  }
}

observations[i].id = nearest.id_i;
```

For each observation, we iterate over each landmark and pick the nearest one to associate with.

### Update weights

Now we can calculate the probability of each particle's location by using the multi-variate Gaussian distribution, comparing the distances between the observed landmarks and the ground truth of the landmark locations. The new particle weight becomes the product of each individual landmark observation probability.

```
for (int j = 0; j < transformed_observations.size(); j++) {
  Map::single_landmark_s nearest_landmark = landmarks[transformed_observations[j].id - 1];
  double gaussian = multi_variate_gaussian(transformed_observations[j].x, transformed_observations[j].y, nearest_landmark.x_f, nearest_landmark.y_f, std_landmark[0], std_landmark[1]);
  
  prob *= gaussian;
}

particles[i].weight = prob;
```

Afterwards, we normalize the particle weights by dividing each weight by the sum of all weights (no exciting code sample to show since it is straightforward division).

### Resample

Now we resample the particles to make sure particles with the highest weight / probability get preserved for the next round. We use C++'s random distribution generator  `std::discrete_distribution` provided as part of the `random` library to accomplish this. `discrete_distribution` generates indices of the highest-quality particles using the weight of each particle to determine its probability of being selected. 

```
for (int i = 0; i < particles.size(); ++i) {
  probabilities.push_back(particles[i].weight);
}

std::default_random_engine generator;
std::discrete_distribution<int> distribution (probabilities.begin(), probabilities.end());

for (int i = 0; i < particles.size(); ++i) {
  int selected_idx = distribution(generator);
  new_particles.push_back(particles[selected_idx]);
}
```

### And that's it

... at least for relevant code that was user-generated for this project ;)
