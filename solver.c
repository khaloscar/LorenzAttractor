// Solver for the lorenz system
// Generates trajectories up to time Tmax, w/ dt timestep
// Lorenz system:
//
//  xdot = omega * (y - x)
//  ydot = rho*x - y -x*z
//  zdot = -beta*z + x*y
//
// pre-compute array size and alloc, then save in one fell swoop?
#include <stdlib.h>
#include <stdio.h>

static double OMEGA, RHO, BETA, DT;

typedef struct {
    double omega;
    double rho;
    double beta;
} LorenzParams;

typedef struct {
    double* t;
    double* x;
    double* y;
    double* z;
} Trajectory;

static void save_solution_step(FILE* file, const char* filename, double t, double x, double y, double z);
static void save_trajectory(const char* filename, Trajectory* trajectory, int n_steps);
// static void parse_input(const int argc, const char* argv[]);
static void lorenz(double x, double y, double z, double* xn, double* yn, double* zn);
static void rk4(Trajectory* state, int i);

int main(const int argc, const char* argv[]) {

    double Tmax = 40;
    double dt = 0.001;

    RHO = 28.0;
    OMEGA = 10.0;
    BETA = 8.0/3.0;

    int n_steps = (int)(Tmax / dt) + 1;
    DT = Tmax/(double)(n_steps - 1);

    Trajectory states;
    states.t = calloc(n_steps, sizeof(double));
    states.x = calloc(n_steps, sizeof(double));
    states.y = calloc(n_steps, sizeof(double));
    states.z = calloc(n_steps, sizeof(double));

    // I.C
    states.t[0] = 0.0;
    states.x[0] = 1.0;
    states.y[0] = 2.0;
    states.z[0] = -4.0;

    for (int ts = 0; ts < n_steps-1; ts++) {
	rk4(&states, ts);
    }

    save_trajectory("test1.txt", &states, n_steps);


    free(states.t);
    free(states.x);
    free(states.y);
    free(states.z);
    return 0;
}

static void save_solution_step(FILE* file, const char* filename, double t, double x, double y, double z) {

    FILE* fp = NULL;
    if (file == NULL) {
	fopen_s(&fp, filename, "a");
    } else { 
	fp = file;
    }

    fprintf_s(fp, "%.16f %.16f %.16f %.16f\n", t, x, y, z); // x, y, z, t

    if (file == NULL) {
	fclose(fp);
    }
}

static void save_trajectory(const char* filename, Trajectory* trajectory, int n_steps) {

    FILE* fp;
    fopen_s(&fp, filename, "w");
    fprintf_s(fp, "time x y z\n");
    for (int i = 0; i < n_steps; i++) {
	save_solution_step(fp, filename,
			    trajectory->t[i], trajectory->x[i],
			    trajectory->y[i], trajectory->z[i]
			    );
    }
    fclose(fp);
}


// static void parse_input(const int argc, const char* argv[]) {
//
//     // -cont FILENAME -> read filename and get then get latest row or something to continue from get go
//     // -omega OMEGA   -> get value for omega
//     // -rho  RHO      -> get value for rho
//     // -beta BETA     -> get value for beta
//     // -xyz X Y Z     -> get value for IC
//
//     //TODO: Fix this shit and figure it out
//
// }

static void rk4(Trajectory* states, int i) {
    // basic RK4 integrator

    double xn = states->x[i];
    double yn = states->y[i];
    double zn = states->z[i];
    double k1x, k1y, k1z;
    lorenz(xn, yn, zn, &k1x, &k1y, &k1z);
    k1x *= DT; k1y *= DT; k1z *= DT;
    double k2x, k2y, k2z;
    lorenz(xn+0.5*k1x, yn+0.5*k1y, zn+0.5*k1z, &k2x, &k2y, &k2z);
    k2x *= DT; k2y *= DT; k2z *= DT;
    double k3x, k3y, k3z;
    lorenz(xn+0.5*k2x, yn+0.5*k2y, zn+0.5*k2z, &k3x, &k3y, &k3z);
    k3x *= DT; k3y *= DT; k3z *= DT;
    double k4x, k4y, k4z;
    lorenz(xn+k3x, yn+k3y, zn+k3z, &k4x, &k4y, &k4z);
    k4x *= DT; k4y *= DT; k4z *= DT;

    states->x[i+1] = xn + ( k1x + 2.0*(k2x + k3x) + k4x) / 6;
    states->y[i+1] = yn + ( k1y + 2.0*(k2y + k3y) + k4y) / 6;
    states->z[i+1] = zn + ( k1z + 2.0*(k2z + k3z) + k4z) / 6;
    states->t[i+1] = states->t[i] + DT;

    // finito
}

static void lorenz(double x, double y, double z,
		   double* fx, double* fy, double* fz) {

    *fx = OMEGA*(y-x);
    *fy = -y + RHO*x - x*z;
    *fz = -BETA*z + x*y; 
    //finito
}
