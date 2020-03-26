// o=========== escape.c ============o
//  agents use reinforcement learning
//  to learn how to escape a small
//  room as quickly as possible.
//  ---------------------------------
//  by s3301419 & s3324818
//  email s3301419@student.rug.nl
// o=================================o

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

// room/agent constraints
enum {
	MAX_ROOM_SIZE = 9,
	MAX_AGENTS = MAX_ROOM_SIZE * MAX_ROOM_SIZE,
	MAX_HEALTH = 2,
};

// what the room can contain
enum {
	FLOOR		= '.',
	WALL		= '=',
	GLASS		= '~',
	SHARDS		= '^',
	DOOR		= 'H',
	OPEN_DOOR	= ']',
	BANDAGE		= '+',
	EXIT		= 'X',
	AGENT		= '@',
};

// misc. constants
enum {
	NONE = -1,
	ESCAPED = 777,
};

typedef enum bool {
	FALSE,
	TRUE
} bool;

// possible actions the agents can take each turn
typedef enum action {
	STAY,
	LEFT,
	RIGHT,
	DOWN,
	UP,
} action;

typedef struct agent {
	int x, y;   // when x or y == ESCAPED, the agent has escaped
	int health; // when health is 0, agent is dead
} agent;

// we use a PCG generator: http://www.pcg-random.org/
// this type holds the state of the RNG, initialize it with seedRNG
typedef uint64_t rng;

int  roomWidth;
int  roomHeight;
char room[MAX_ROOM_SIZE][MAX_ROOM_SIZE];
int numAgents;
agent agents[MAX_AGENTS];

// we store a copy at the room when running an epoch
// so that we can "reset" to the original configuration
// when the epoch ends by copying it back
char backupRoom[sizeof(room)];
char backupAgents[sizeof(agents)];

// dimensions of the Q-table:
//   2   - we need 2 tables for double Q
// (9x9) - agent position in the room
//   2   - 2 or 1 health
// (3^8) - each agent sees 8 cells and each cell can have 3 state
//   5   - number of actions the agent can take
// = 10,628,820 entries (1,062,882 states)
double qTable[2][MAX_ROOM_SIZE][MAX_ROOM_SIZE][MAX_HEALTH][3][3][3][3][3][3][3][3][5];

double totalReward; // total reward obtained by ALL agents combined over 1 epoch
rng globalRNG;
int currEpoch;
int currTurn;
int maxSteps = 200; // how many turns to do per epoch
bool printEpochs = TRUE; // if TRUE, then results are printed to console after every epoch

// Q learning parameters
double alpha = 0.5;
double gamma = 0.95;
double epsilon = 0.05;
double optimism = 50;
double escapeReward	   = +1000;
double deathPunishment = -1000;
double idlePunishment  = -1;
bool useDoubleQ = FALSE; // if TRUE, then use double Q-learning
bool useEpsilon = TRUE; // if TRUE, then use epsilon greedy, otherwise just use greedy

FILE *resultsFile; // store results in this file

// initialize the PCG RNG with a seed
rng seedRNG(int seed) {
	return
		((uint32_t)seed + 1442695040888963407u)
		* 6364136223846793005u
		+ 1442695040888963407u;
}

// get random float in [0,1]
double randf(rng *rng) {
	uint64_t x = *rng;
	uint32_t r = (uint32_t)(x >> 59);
	*rng = x
		* 6364136223846793005u
		+ 1442695040888963407u;

	x ^= x >> 18;
	uint32_t y = (uint32_t)(x >> 27);
	y = y >> r | y << ((uint32_t)(-(int)r) & 31);

	return y / (1.0 + UINT_MAX);
}

// get random agent action
action randAction(rng *rng) {
	return (action)(randf(rng) * (1.0 + UP));
}

// clamp x between min and max
int clamp(int x, int min, int max) {
	return
		x < min ? min :
		x > max ? max : x;
}

// return TRUE if (x,y) is inside of the room dimensions
bool isInRoom(int x, int y) {
	return
		x >= 0 && x < roomWidth &&
		y >= 0 && y < roomHeight;
}

// returns the index of the agent at (x,y) or NONE if none is there
int agentAt(int x, int y) {
	if (isInRoom(x, y)) {
		for (int a = 0; a < numAgents; ++a) {
			if (agents[a].x == x && agents[a].y == y) {
				return a;
			}
		}
	}
	return NONE;
}

// return TRUE if agent could stand on the given cell
// for example, agents can stand on a floor or an open door
// but not on a wall, or a closed door
bool isPassable(char cell) {
	switch (cell) {
		case FLOOR:
		case SHARDS:
		case OPEN_DOOR:
		case BANDAGE:
		case EXIT:
			return TRUE;
		case WALL:
		case GLASS:
		case DOOR:
		default:
			return FALSE;
	}
}

// load room configuration from given file
// or load empty 9x9 room in case of error
void loadRoom(const char *filename) {
	printf("loading %s ... ", filename);
	FILE *roomFile = fopen(filename, "rt");
	if (roomFile != NULL) {
		roomWidth = 0;
		roomHeight = 0;
		numAgents = 0;
		int c, x = 0, y = 0;
		do {
			c = getc(roomFile);
			if (c == '\n' || c == '\r' || c == EOF) {
				if (x > 0) {
					// start new row
					if (roomWidth == 0) {
						roomWidth = x;
					}

					if (x != roomWidth) {
						printf("inconsistent room dimensions\n");
						goto makeDefaultRoom;
					} else if (y >= MAX_ROOM_SIZE){
						printf("room too tall\n");
						goto makeDefaultRoom;
					}

					x = 0;
					++y;
				}
			} else {
				// add new column
				if (roomWidth > 0 && x >= roomWidth) {
					printf("inconsistent room dimensions\n");
					goto makeDefaultRoom;
				} else if (x >= MAX_ROOM_SIZE) {
					printf("room too wide\n");
					goto makeDefaultRoom;
				}

				if (c == AGENT) {
					// add new agent
					if (numAgents < MAX_AGENTS) {
						room[x][y] = FLOOR;
						agents[numAgents].health = 2;
						agents[numAgents].x = x;
						agents[numAgents].y = y;
						++numAgents;
					} else {
						printf("too many agents specified\n");
						goto makeDefaultRoom;
					}
				} else {
					// add new cell
					room[x][y] = (char)c;
				}

				++x;
			}
		} while (c != EOF);

		roomHeight = y;
		if (roomWidth < 1 || roomWidth > MAX_ROOM_SIZE) {
			printf("room too wide\n");
			goto makeDefaultRoom;
		} else if (roomHeight < 1 || roomHeight > MAX_ROOM_SIZE) {
			printf("room too tall\n");
			goto makeDefaultRoom;
		}

		// flip room horizontally since we read it in backwards
		for (x = 0; x < roomWidth; ++x) {
			for (y = 0; y < roomHeight / 2; ++y) {
				char temp = room[x][y];
				room[x][y] = room[x][roomHeight - y - 1];
				room[x][roomHeight - y - 1] = temp;
			}
		}

		// also flip all the agents
		for (int agent = 0; agent < numAgents; ++agent) {
			agents[agent].y = roomHeight - agents[agent].y - 1;
		}

		printf("done\n");
	} else {
		printf("file not found\n");

	makeDefaultRoom:
		numAgents = 0;
		roomWidth = 9;
		roomHeight = 9;
		for (int x = 0; x < roomWidth; ++x) {
			for (int y = 0; y < roomHeight; ++y) {
				room[x][y] = FLOOR;
			}
		}

		printf("loaded default 9x9 room\n");
	}

	if (roomFile != NULL) {
		fclose(roomFile);
	}
}

// load all Q-table with an initial value
void loadQTable(double initialValues) {
	optimism = initialValues;
	double *q = &qTable[0][0][0][0][0][0][0][0][0][0][0][0][0];
	double numEntries = sizeof(qTable) / sizeof(*q);
	for (int i = 0; i < numEntries; ++i) {
		q[i] = optimism;
	}
}

// open a file to which results from every epoch will be stored
// note that the entire file will be cleared
void openResultsFile(const char *filename) {
	if (resultsFile != NULL) {
		fclose(resultsFile);
	}

	resultsFile = fopen(filename, "r");
	if (resultsFile == NULL) {
		printf("creating %s ... ", filename);
	} else {
		fclose(resultsFile);
		printf("clearing %s ... ", filename);
	}

	resultsFile = fopen(filename, "wt");
	if (resultsFile != NULL) {
		fprintf(resultsFile, "epoch, total reward\n");
		printf("done\n");
	} else {
		printf("couldn't open file\n");
	}
}

// get the Q-table entries for both Q-tables for the given
// agent and using the current state (room and agents)
// *qA and *qB will point into the position of the entry for
// the FIRST of FIVE actions the agent can take in this state
void getQEntry(int agent, double **qA, double **qB) {
	assert(agents[agent].health > 0 && agents[agent].health <= MAX_HEALTH);
	assert(isInRoom(agents[agent].x, agents[agent].y));

	int x  = agents[agent].x;
	int y  = agents[agent].y;
	int hp = agents[agent].health;

	// the agents can see cells around them in a crosshair:
	//       [ ]
	//       [ ]
	// [ ][ ] @ [ ][ ]
	//       [ ]
	//       [ ]
	// the arrays below store corrdinate offsets of the visible cells
	const int xOffsets[] = { -2, -1, 1, 2,  0,  0, 0, 0 };
	const int yOffsets[] = {  0,  0, 0, 0, -2, -1, 1, 2 };

	// each visible cell can be either:
	enum {
		DEACTIVATED, // e.g. FLOOR (static) or DOOR (dynamic)
		ACTIVATED,   // OPEN_DOOR, BROKEN_GLASS, and BANDAGE
		HAS_AGENT    // overrides all other cell states above
	} state[8];

	// we need to check for all visible cells if they contain an agent
	// but agentAt loops over all agents - by precomputing which cells
	// have an agent we can avoid O(N^2) performance
	// we REALLY should have done this globally for maximum performance
	// but that would have made everything too complicated
	assert(MAX_ROOM_SIZE < 8 * sizeof(int));
	int occupancy[MAX_ROOM_SIZE];
	memset(occupancy, 0, sizeof(occupancy));
	for (int a = 0; a < numAgents; ++a) {
		if (isInRoom(agents[a].x, agents[a].y)) {
			occupancy[agents[a].x] |= (1 << (agents[a].y));
		}
	}

	// loop over visible cells
	for (int vision = 0; vision < 8; ++vision) {
		int cx = x + xOffsets[vision];
		int cy = y + yOffsets[vision];

		if (isInRoom(cx, cy)) {
			if (occupancy[cx] & (1 << cy)) {
				state[vision] = HAS_AGENT;
			} else {
				state[vision] =
					room[cx][cy] == SHARDS    ? ACTIVATED :
					room[cx][cy] == OPEN_DOOR ? ACTIVATED :
					room[cx][cy] == BANDAGE   ? ACTIVATED :
					DEACTIVATED;
			}
		} else {
			state[vision] = DEACTIVATED;
		}
	}

	*qA =
		qTable[0][x][y][hp - 1]
		[state[0]][state[1]][state[2]][state[3]]
		[state[4]][state[5]][state[6]][state[7]];
	*qB = !useDoubleQ ? NULL :
		qTable[1][x][y][hp - 1]
		[state[0]][state[1]][state[2]][state[3]]
		[state[4]][state[5]][state[6]][state[7]];
}

// loop through all possible actions and find the best one:
// if qB is NULL, the action with highest qA is returned
// otherwise, the action with the highest average of qA and qB is returned
action getBestAction(double *qA, double *qB) {
	action maxa = STAY;
	double maxq = -INFINITY;

	for (action a = STAY; a <= UP; ++a) {
		if (qA[a] + (qB ? qB[a] : 0) > maxq) {
			maxq = qA[a] + (qB ? qB[a] : 0);
			maxa = a;
		}
	}

	return maxa;
}

// modify (*x,*y) according to the given action
void actionModCoords(action a, int *x, int *y) {
	switch (a) {
		case LEFT:	*x -= 1; break;
		case RIGHT: *x += 1; break;
		case DOWN:	*y -= 1; break;
		case UP:	*y += 1; break;
		default: /* STAY */ break;
	}

	*x = clamp(*x, 0, roomWidth  - 1);
	*y = clamp(*y, 0, roomHeight - 1);
}

// simulate an entire turn of agents escaping
// return TRUE if an epoch has passed after the rurn
// this is where the interesting stuff is!
bool simulateTurn() {
	if (currTurn == 0) {
		// make a backup of the room before changing anything!
		memcpy(backupRoom, room, sizeof(room));
		memcpy(backupAgents, agents, sizeof(agents));
	}

	// various things about the agent's decision is stored here
	struct actionrecord {
		bool isEscaping; // is the agent alive and not escaped yet?
		int x, y;        // position before moving
		int dx, dy;      // position to which the agent wants to move
		action action;   // action that the agent picked
		double *q0, *q1; // pointers into the Q-table for the state before any action is taken
	} actionRecords[MAX_AGENTS];

	// when checking for collisions we will frequently want to know which agent
	// wants to move where - rather than looping through all the agents we use a
	// map which stores at each (x,y) which agent wants to move there (or NONE)
	int collisionMap[MAX_ROOM_SIZE][MAX_ROOM_SIZE];
	bool someAgentsAreEscaping = FALSE; // if everybody escapes we can immediately start the next epoch

	memset(actionRecords, 0, sizeof(actionRecords));
	memset(collisionMap, NONE, sizeof(collisionMap)); // this DOES work because NONE == -1 == 0xFFF..

	// decide action and resolve collisions for each agent
	for (int a = 0; a < numAgents; ++a) {
		// initialize the record
		int x = agents[a].x;
		int y = agents[a].y;
		struct actionrecord *record = &actionRecords[a];
		record->x = x;
		record->y = y;
		record->dx = x;
		record->dy = y;
		if (isInRoom(x, y)) {
			if (agents[a].health > 0) {
				record->isEscaping    = TRUE;
				someAgentsAreEscaping = TRUE;

				// get action based on policy
				double *q0, *q1;
				getQEntry(a, &q0, &q1);
				action act;
				if (useEpsilon && randf(&globalRNG) < epsilon) {
					act = randAction(&globalRNG); // epsilon
				} else {
					act = getBestAction(q0, q1);  // greedy
				}

				// store Q-entries so we can update them later
				actionRecords[a].q0 = q0;
				actionRecords[a].q1 = q1;
				actionRecords[a].action = act;

				actionModCoords(act, &x, &y);
				if (isPassable(room[x][y])) {
					// agent can move here
					record->dx = x;
					record->dy = y;
				} else {
					// agent can't move here
					x = record->x;
					y = record->y;
				}

				// resolve collisions with other agents by looking up
				// the collision map - agents that collide stay in place
				int b = collisionMap[x][y];
				if (b != NONE) {
					// collision a->b !
					// 1. stop b from moving
					// 2. stop a from moving
					do {
						// 1.1. b moves to where it started the turn
						struct actionrecord *brec = &actionRecords[b];
						brec->dx = brec->x;
						brec->dy = brec->y;

						// 1.2. check for new collisions b'->b
						int next = collisionMap[brec->x][brec->y];
						collisionMap[brec->x][brec->y] = b;
						if (next == b) { // if b chose to STAY the chain is broken
							next = NONE;
						}
						b = next; // continue running down the collision chain
					} while (b != NONE);

					// 2.1. a moves to where it started the turn
					x = record->x;
					y = record->y;
					record->dx = x;
					record->dy = y;

					// 2.2. check for new collisions with a
					int c = collisionMap[x][y];
					while (c != NONE) {
						// collision c->a !
						// 2.3. c moves to where it started the turn
						struct actionrecord *crec = &actionRecords[c];
						crec->dx = crec->x;
						crec->dy = crec->y;
						// 2.4. check for new collisions c'->c
						int next = collisionMap[crec->x][crec->y];
						collisionMap[crec->x][crec->y] = c;
						if (next == c) {
							next = NONE;
						}
						c = next; // continue running down the collision chain
					}
				}
				collisionMap[x][y] = a;
			}
		}
	}

	// act on decision
	for (int a = 0; a < numAgents; ++a) {
		if (actionRecords[a].isEscaping) {
			int  x = actionRecords[a].x;
			int  y = actionRecords[a].y;
			int dx = actionRecords[a].dx;
			int dy = actionRecords[a].dy;
			action act = actionRecords[a].action;

			// if agent chose to move, but didn't, it might be
			// because it moved onto a door and so should open it
			if (act != STAY && x == dx && y == dy) {
				actionModCoords(act, &x, &y);
				if (room[x][y] == GLASS) {
					room[x][y] = SHARDS;
				} else if (room[x][y] == DOOR) {
					room[x][y] = OPEN_DOOR;
				}
			} else {
				assert(isPassable(room[dx][dy]));
				agents[a].x = dx;
				agents[a].y = dy;
			}
		}
	}

	// get reward and learn from decision
	for (int a = 0; a < numAgents; ++a) {
		if (actionRecords[a].isEscaping) {
			int x = agents[a].x;
			int y = agents[a].y;
			action act = actionRecords[a].action;

			// assign rewards and determine if state is terminal
			double reward = idlePunishment;
			bool isTerminalState = FALSE;

			if (room[x][y] == EXIT) {
				agents[a].x = ESCAPED;
				agents[a].y = ESCAPED;
				isTerminalState = TRUE;
				reward = escapeReward;
			} else if (room[x][y] == SHARDS) {
				agents[a].health -= 1;
				if (agents[a].health == 0) {
					isTerminalState = TRUE; // agent died
					reward = deathPunishment;
				}
			} else if (room[x][y] == BANDAGE) {
				room[x][y] = FLOOR;
				if (agents[a].health < MAX_HEALTH) {
					agents[a].health = MAX_HEALTH;
				}
			}

			totalReward += reward;

			if (useDoubleQ) {
				double *q00 = &actionRecords[a].q0[act];
				double *q10 = &actionRecords[a].q1[act];
				double q01 = 0, q11 = 0; // Q[terminal-state] = 0
				if (!isTerminalState) {
					double *q01p, *q11p;
					getQEntry(a, &q01p, &q11p);
					q01 = *(q01p + getBestAction(q11p, NULL));
					q11 = *(q11p + getBestAction(q01p, NULL));
				}

				// update only 1 Q-table at random
				if (randf(&globalRNG) < 0.5) {
					*q00 += alpha * (reward + gamma * q11 - (*q00));
				} else {
					*q10 += alpha * (reward + gamma * q01 - (*q10));
				}
			} else {
				double *q0 = &actionRecords[a].q0[act];
				double q1 = 0; // Q[terminal-state] = 0
				if (!isTerminalState) {
					double *qA, *qB;
					getQEntry(a, &qA, &qB);
					q1 = *(qA + getBestAction(qA, NULL));
				}

				*q0 += alpha * (reward + gamma * q1 - (*q0));
			}
		}
	}

	if (++currTurn >= maxSteps || !someAgentsAreEscaping) {
		// epoch ended - print the results
		if (printEpochs) {
			printf("epoch %d: RT = %lg\n", 1 + currEpoch, totalReward);
		}
		if (resultsFile != NULL) {
			fprintf(resultsFile, "%d, %lg\n", currEpoch, totalReward);
		}

		// restore all backups
		++currEpoch;
		currTurn    = 0;
		totalReward = 0;
		memcpy(room, backupRoom, sizeof(room));
		memcpy(agents, backupAgents, sizeof(agents));

		return TRUE;
	}

	return FALSE;
}

//           __
//           ||
// o====================o
// <----- CLI Code ----->
// o====================o
//           ||
//           ||
//           ||
//           ||
//           ||
//           ||
//__________****__________

// print CLI help message to console
void printCLIHelp() {
	printf("o=============== Escape Room ===============o\n");
	printf("  enter any of these commands at the prompt\n");
	printf(" -------------------------------------------\n");
	printf(" h|help        print this help message\n");
	printf(" q|quit        quit the program\n");
	printf(" r|room        print room\n");
	printf(" e|epochs [N]  advance N epochs (default=1)\n");
	printf(" t|turns [N]   advance N turns (default=1)\n");
	printf(" s|seed N      seed the RNG\n");
	printf(" alpha X       set alpha to X\n");
	printf(" gamma X       set gamma to X\n");
	printf(" epsilon X     set epsilon to X\n");
	printf(" setq X        set Q-values to X\n");
	printf(" doubleq 1|0   toggle double Q-learning\n");
	printf(" load F        load room file F\n");
	printf(" saveto F      save results to file F\n");
	printf(" reproduce     get results used in the paper\n");
	printf("o===========================================o\n");
}

// search the string for the first instance where search(*string) returns TRUE
// for example searchFor(isspace, "Foo Bar") will find the first space and return " Bar"
const char *searchFor(int(*search)(int chr), const char *string) {
	while (*string != 0 && !search(*string)) {
		++string;
	}
	return string;
}

// check if cmd is target
bool cmdIs(const char *target, const char *cmd) {
	int i;
	for (i = 0; target[i]; ++i) {
		if (tolower(cmd[i]) != tolower(target[i])) {
			return FALSE;
		}
	}
	return !isgraph(cmd[i]);
}

// run the given command
// check printCLIHelp for a list of commands
void runCmd(const char *command) {
	char cmdcopy[256];
	strncpy(cmdcopy, command, 255);
	cmdcopy[255] = 0;
	for (int i = 0; cmdcopy[i]; ++i) {
		if (cmdcopy[i] == '\n' || cmdcopy[i] == '\r') {
			cmdcopy[i] = 0;
		}
	}

	const char *cmd    = searchFor(isgraph, cmdcopy);
	const char *cmdEnd = searchFor(isspace, cmd    );
	const char *arg    = searchFor(isgraph, cmdEnd );
	const char *argEnd = searchFor(isspace, arg    );
	const char *arg2   = searchFor(isgraph, argEnd );

	if (*arg2) {
		printf("excessive argument '%s'\n", arg2);
		return;
	}

	if (cmdIs("help", cmd) || cmdIs("h", cmd)) {
		if (!*arg) {
			printCLIHelp();
		} else {
			printf("excessive argument '%s'\n", arg);
		}
	} else if (cmdIs("quit", cmd) || cmdIs("q", cmd) || cmdIs("exit", cmd)) {
		if (!*arg) {
			if (resultsFile != NULL) {
				fclose(resultsFile);
			}
			exit(0);
		} else {
			printf("excessive argument '%s'\n", arg);
		}
	} else if (cmdIs("room", cmd) || cmdIs("r", cmd)) {
		if (!*arg) {
			for (int y = roomHeight - 1; y >= 0; --y) {
				for (int x = 0; x < roomWidth; ++x) {
					int agent = agentAt(x, y);
					if (agent != NONE) {
						int hp = agents[agent].health;
						if (hp == MAX_HEALTH) {
							putchar('@');
						} else if (hp > 0) {
							putchar('Q');
						} else {
							putchar('x');
						}
					} else {
						putchar(room[x][y]);
					}
				}
				putchar('\n');
			}
		} else {
			printf("excessive argument '%s'\n", arg);
		}
	} else if (cmdIs("epochs", cmd) || cmdIs("e", cmd)) {
		int numEpochs;
		if (sscanf(arg, "%d", &numEpochs) != 1) {
			numEpochs = 1;
		}
		for (int epoch = 0; epoch < numEpochs; epoch += simulateTurn());
	} else if (cmdIs("turns", cmd) || cmdIs("t", cmd)) {
		int numTurns;
		if (sscanf(arg, "%d", &numTurns) != 1) {
			numTurns = 1;
		}
		for (int turn = 1; turn < numTurns; ++turn) {
			simulateTurn();
		}
	} else if (cmdIs("seed", cmd) || cmdIs("s", cmd)) {
		int seed;
		if (sscanf(arg, "%d", &seed) == 1) {
			globalRNG = seedRNG(seed);
		} else {
			printf("missing argument N\n");
		}
	} else if (cmdIs("alpha", cmd)) {
		double a;
		if (sscanf(arg, "%lf", &a) == 1) {
			if (a >= 0 && a <= 1) {
				alpha = a;
			} else {
				printf("invalid argument X: must be in [0,1]\n");
			}
		} else {
			printf("alpha = %lg\n", alpha);
		}
	} else if (cmdIs("gamma", cmd)) {
		double g;
		if (sscanf(arg, "%lf", &g) == 1) {
			if (g >= 0 && g <= 1) {
				gamma = g;
			} else {
				printf("invalid argument X: must be in [0,1]\n");
			}
		} else {
			printf("gamma = %lg\n", gamma);
		}
	} else if (cmdIs("epsilon", cmd)) {
		double e;
		if (sscanf(arg, "%lf", &e) == 1) {
			if (e >= 0 && e <= 1) {
				epsilon = e;
			} else {
				printf("invalid argument X: must be in [0,1]\n");
			}
		} else {
			printf("epsilon = %lg\n", epsilon);
		}
	} else if (cmdIs("setq", cmd)) {
		double qValues;
		if (sscanf(arg, "%lf", &qValues) == 1) {
			loadQTable(qValues);
			currEpoch = 0;
		} else {
			printf("optimism = %lg\n", optimism);
		}
	} else if (cmdIs("doubleq", cmd) || cmdIs("dq", cmd)) {
		bool doubleQ;
		if (sscanf(arg, "%d", &doubleQ) == 1) {
			if (doubleQ == 0 || doubleQ == 1) {
				useDoubleQ = doubleQ;
			} else {
				printf("invalid argument: must be 0 or 1\n");
			}
		} else {
			printf("double Q-learning is %s\n", useDoubleQ ? "on" : "off");
		}
	} else if (cmdIs("load", cmd) || cmdIs("loadr", cmd)) {
		if (*arg != 0) {
			loadRoom(arg);
		} else {
			printf("missing argument F\n");
		}
	} else if (cmdIs("saveto", cmd)) {
		if (*arg != 0) {
			openResultsFile(arg);
		} else {
			printf("missing argument F\n");
		}
	} else if (cmdIs("reproduce", cmd)) {
		if (!*arg) {
			int numRuns = 200;
			printEpochs = FALSE;
			printf("reproducing paper results ... this may take up to 10 minutes\n");
			runCmd("epsilon 0.005");

			runCmd("doubleq 0");
			runCmd("alpha 0.2");
			runCmd("gamma 0.9");
			{
				runCmd("seed 42");
				runCmd("load room1.txt");
				runCmd("saveto results1.csv");
				printf("reproducing room1 ");
				for (int run = 0; run < numRuns; ++run) {
					runCmd("setq 100");
					runCmd("epochs 3000");
					if ((run + 1) % (numRuns / 3) == 0) {
						printf(".");
					}
				}
				printf(" done\n");

				runCmd("seed 42");
				runCmd("load room2.txt");
				runCmd("saveto results2.csv");
				printf("reproducing room2 ");
				for (int run = 0; run < numRuns; ++run) {
					runCmd("setq 100");
					runCmd("epochs 3000");
					if ((run + 1) % (numRuns / 3) == 0) {
						printf(".");
					}
				}
				printf(" done\n");

				runCmd("seed 42");
				runCmd("load room3.txt");
				runCmd("saveto results3.csv");
				printf("reproducing room3 ");
				for (int run = 0; run < numRuns; ++run) {
					runCmd("setq 100");
					runCmd("epochs 3000");
					if ((run + 1) % (numRuns / 3) == 0) {
						printf(".");
					}
				}
				printf(" done\n");
			}

			runCmd("doubleq 1");
			{
				runCmd("seed 42");
				runCmd("alpha 0.2");
				runCmd("gamma 0.9");
				runCmd("load room1.txt");
				runCmd("saveto results1d.csv");
				printf("reproducing room1 (double Q) ");
				for (int run = 0; run < numRuns; ++run) {
					runCmd("setq 50");
					runCmd("epochs 3000");
					if ((run + 1) % (numRuns / 3) == 0) {
						printf(".");
					}
				}
				printf(" done\n");

				runCmd("seed 42");
				runCmd("alpha 0.3");
				runCmd("gamma 0.8");
				runCmd("load room2.txt");
				runCmd("saveto results2d.csv");
				printf("reproducing room2 (double Q) ");
				for (int run = 0; run < numRuns; ++run) {
					runCmd("setq 50");
					runCmd("epochs 3000");
					if ((run + 1) % (numRuns / 3) == 0) {
						printf(".");
					}
				}
				printf(" done\n");

				runCmd("seed 42");
				runCmd("alpha 0.15");
				runCmd("gamma 0.8");
				runCmd("load room3.txt");
				runCmd("saveto results3d.csv");
				printf("reproducing room3 (double Q) ");
				for (int run = 0; run < numRuns; ++run) {
					runCmd("setq 50");
					runCmd("epochs 3000");
					if ((run + 1) % (numRuns / 3) == 0) {
						printf(".");
					}
				}
				printf(" done\n");
			}

			runCmd("saveto results_.csv");
			printf("reproduction complete :)\n");
			printEpochs = TRUE;
		} else {
			printf("excessive argument '%s'\n", arg);
		}
	} else if (strlen(cmd) > 0) {
		printf("unknown command '%s'\n", cmdcopy);
	}
}

// run the CLI - get user input in a loop, etc.
void runCLI() {
	printf("\no========= Escape Room =========o\n");
	printf(" type 'help' for a commands list\n");
	printf("o===============================o\n\n");
	for (;;) {
		printf(">>> ");
		char input[256];
		if (fgets(input, sizeof(input), stdin) != NULL) {
			runCmd(input);
		}
	}
}

//           __
//           ||
// o====================o
// <----- GUI Code ----->
// o====================o
//           ||
//           ||
//           ||
//           ||
//           ||
//           ||
//__________****__________
//
// this portion of the code draws and runs the GUI
// it isnt pretty but it works
// if you dont want to compile with the GUI,
// just: #define NOGUI

#ifndef NOGUI

#define GLFW_INCLUDE_NONE
#include "glad.h"
#include "glfw3.h"

const double pi = 3.141592653589793;

// UI parameters
enum {
	MAX_UNDO = 100000,
};

// UI states
typedef enum  {
	EDITING,
	RUNNING,
	PAUSED,
} state;

// records the type of change, see performAction
typedef enum {
	// replaces a cell at given coordinates
	REPLACE_CELL = 1,
	// insert agent into specified index and coordinates
	//@NOTE: currently there is no way to set the health
	INSERT_AGENT,
	// removes an agent from a specified index
	REMOVE_AGENT,
	// resize the room to specified width and height
	RESIZE_ROOM,
} change;

// stores a record of changes made through
// the UI so that we can undo/redo them
typedef struct {
	change change;	// which action was performed
	int groupSize;	// how many actions to undo/redo
	union {
		// only valid if change == REPLACE_CELL
		struct {
			int x, y;		// cell coordinates of cell
			char newCell;	// cell which replaced the old cell
			char oldCell;	// cell which was replaced
		} replaceCell;

		// only valid if change == INSERT_AGENT
		struct {
			int x, y;		// cell coords of new agent
			int agentIndex;	// index of agent in the global agent list
			int agentHealth;// how much health the agent had
		} insertAgent;

		// only valid if change == REMOVE_AGENT
		struct {
			int x, y;		// cell coords where agent was replaced
			int agentIndex;	// index of agent in the global agent list
			int agentHealth;// how much health the agent had
		} removeAgent;

		// only valid if change == RESIZE_ROOM
		struct {
			int newWidth, newHeight;	// new size of room in cells
			int oldWidth, oldHeight;	// old size of room in cells
		} resizeRoom;
	};
} changeinfo;

// 8-BPC color
typedef uint32_t rgba;

// this stuff automatically gets updated when
// the window events fire, it should all be
// considered READ ONLY
GLFWwindow *window = NULL;		// window handle
int windowWidth, windowHeight;	// window size in pixels
int mouseX, mouseY;				// position in screen coords
bool dragging;		            // true when the mouse is being dragged
int draggedAgent = NONE;	    // which agent the mouse is moving, or NONE
char selectedCell = NONE;	    // which cell to place with left click
int visualizeQTable;            // if not 0, then highlight each cell with values from the respective Q-entries
bool grayscale;				    // if TRUE, draw everything in black and white (for better Q-table visualization)
state uiState = EDITING;	    // current state of the GUI
double turnFreq = 1;            // number of turns to (try to) run per second
bool fastMode;				    // in "fast mode" 10000 turns are simulated each frame

// to allow for arbitrary panning & zooming
// we transform the position of every cell
// before drawing: pos = translate(scale(pos))
double transX, transY;	// translation in pixels
double scale;			// scale part of transform

// for undo/redo we store two indices into the
// global change stack: the undo top (UT, UndoT)
// and the redo top (RT, RedoT)
// these are used like in the following diagram
// ---------------------------------------------
//
// [start] change0 change1 [empty]
//                         ^UT&RT^
// == +change2 =>
//
// [start] change0 change1 change2 [empty]
//                                 ^UT&RT^
// == 3 x undo =>
//
// [start] change0 change1 change2 [empty]
//         ^UndoT^                 ^RedoT^
// == redo =>
//
// [start] change0 change1 change2 [empty]
//                 ^UndoT^         ^RedoT^
// == +change3 =>
//
// [start] change0 change3 [empty]
//                         ^UT&RT^
changeinfo changeStack[MAX_UNDO];
int redoTop;
int undoTop;

// print GUI control help message
void printGUIHelp() {
	printf("\no================ Escape Room ================o\n");
	printf("  controls\n");
	printf(" ------------- general controls --------------\n");
	printf(" h              show this help message\n");
	printf(" x              enter CLI command\n");
	printf(" c              recenter camera\n");
	printf(" s              print room size\n");
	printf(" scroll         zoom (nothing selected)\n");
	printf(" mouse drag     pan screen (nothing selected)\n");
	printf(" alt+scroll     zoom\n");
	printf(" alt+drag       pan screen\n");
	printf(" q              reset Q-values\n");
	printf(" e              toggle epsilon-greedy\n");
	printf(" v              visualize Q table\n");
	printf(" f              superfast mode\n");
	printf(" >              step 1 turn\n");
	printf(" ------------- editing controls --------------\n");
	printf(" 1              select 'wall'\n");
	printf(" 2              select 'glass'\n");
	printf(" 3              select 'door'\n");
	printf(" 4              select 'exit'\n");
	printf(" 5              select 'agent'\n");
	printf(" scroll         cycle through selections\n");
	printf(" middle click   select clicked cell\n");
	printf(" ctrl+z         undo\n");
	printf(" ctrl+y         redo\n");
	printf(" arrow keys     resize the room\n");
	printf(" mouse drag     move agent (agent selected)\n");
	printf(" left click     place cell\n");
	printf(" left click     break glass or open doors\n");
	printf(" right click    remove agent (agent selected)\n");
	printf(" right click    remove cell\n");
	printf(" enter/space    start running\n");
	printf(" ------------- running/paused controls --------------\n");
	printf(" space          pause/unpause\n");
	printf(" enter          go back to editing\n");
	printf(" 0-9            change simulation speed\n");
	printf("o=============================================o\n\n");
}

// clamp x between min and max
double clampf(double x, double min, double max) {
	return
		x < min ? min :
		x > max ? max : x;
}

// insert new agent at specified index and (x,y)
void insertAgent(int index, int x, int y) {
	assert(index >= 0 && index <= numAgents);
	assert(numAgents < MAX_AGENTS);
	if (index != numAgents) {
		memmove(
			&agents[index + 1],
			&agents[index],
			(numAgents - index) * sizeof(*agents));
	}
	++numAgents;

	agents[index].x = x;
	agents[index].y = y;
	agents[index].health = MAX_HEALTH;
}

// remove agent at specified index
void removeAgent(int index) {
	assert(index >= 0 && index < numAgents);
	--numAgents;
	if (index != numAgents) {
		memmove(
			&agents[index],
			&agents[index + 1],
			(numAgents - index) * sizeof(*agents));
	}
}

// return the cell as a string so that we can print it to the user
const char *toString(char cell) {
	switch (cell) {
		case FLOOR:		return "Floor";
		case WALL:		return "Wall";
		case GLASS:		return "Glass";
		case SHARDS:	return "Shards";
		case DOOR:		return "Closed Door";
		case OPEN_DOOR:	return "Open Door";
		case BANDAGE:	return "Bandage";
		case EXIT:		return "Exit";
		case AGENT:		return "Agent";
		default:		return "Void";
	}
}

// use this to change anything in the room if you want changes to work with undo/redo
//  performChange(REPLACE_CELL, x, y, FLOOR, 1)
//  performChange(INSERT_AGENT, x, y, index, 1)
//  performChange(REMOVE_AGENT, 0, 0, index, 1)
//  performChange(RESIZE_ROOM, w, h, 0, 1)
// these are meant to be as general as possible because we need
// to add a lot of code (3 functions) for every possible change
bool performChange(change ch, int x, int y, int cellOrAgent, int groupSize) {
	if (uiState == EDITING) {
		// check whether this action actually
		// changes anything before adding it
		// to the undo stack
		bool commit = FALSE;
		changeinfo change;
		change.change = ch;
		change.groupSize = groupSize;

		assert(groupSize > 0);

		switch (ch) {
			case REPLACE_CELL: {
				// check if cell is inside the room
				if (isInRoom(x, y)) {
					char newCell = (char)cellOrAgent;
					char oldCell = room[x][y];

					// check if new cell is actually different
					if (oldCell != newCell) {
						// we cant place an unpassable
						// cell on top of an agent
						if (isPassable(newCell) || agentAt(x, y) < 0) {
							commit = TRUE;
							change.replaceCell.x = x;
							change.replaceCell.y = y;
							change.replaceCell.oldCell = oldCell;
							change.replaceCell.newCell = newCell;
							room[x][y] = newCell;
						}
					}
				}
			} break;
			case INSERT_AGENT: {
				// check if new agent is being placed inside the room
				if (isInRoom(x, y)) {
					int agent = cellOrAgent;
					// check if index is valid
					if (agent >= 0 && agent <= numAgents) {
						// we cant place an agent on an unpassable cell
						if (agentAt(x, y) < 0 && isPassable(room[x][y])) {
							commit = TRUE;
							change.insertAgent.x = x;
							change.insertAgent.y = y;
							change.insertAgent.agentIndex = agent;
							change.insertAgent.agentHealth = MAX_HEALTH;
							insertAgent(agent, x, y);
						}
					}
				}
			} break;
			case REMOVE_AGENT: {
				int agent = cellOrAgent;
				// check if index if valid
				if (agent >= 0 && agent < numAgents) {
					commit = TRUE;
					change.insertAgent.x = agents[agent].x;
					change.insertAgent.y = agents[agent].y;
					change.insertAgent.agentIndex = agent;
					change.insertAgent.agentHealth = agents[agent].health;
					removeAgent(agent);
				}
			} break;
			case RESIZE_ROOM: {
				// check if new size is valid and different from old size
				if (x > 0 && x <= MAX_ROOM_SIZE &&
					y > 0 && y <= MAX_ROOM_SIZE &&
					(x != roomWidth || y != roomHeight)) {
					commit = TRUE;
					assert(change.groupSize == 1);
					change.resizeRoom.newWidth = x;
					change.resizeRoom.newHeight = y;
					change.resizeRoom.oldWidth = roomWidth;
					change.resizeRoom.oldHeight = roomHeight;

					// if the new size is smaller then we need
					// remove all agents and replace all cells
					// that are being cut off and place them
					// in the same change group
					if (x < roomWidth || y < roomHeight) {
						int numChanges = 0;

						// remove all cells outside new room dimensions
						for (int cx = 0; cx < roomWidth; ++cx) {
							for (int cy = 0; cy < roomHeight; ++cy) {
								if ((cx >= x || cy >= y) && room[cx][cy] != FLOOR) {
									++numChanges;
									bool success = performChange(REPLACE_CELL, cx, cy, FLOOR, 1);
									assert(success);
								}
							}
						}

						// then remove agents
						for (int a = 0; a < numAgents; ++a) {
							if (agents[a].x >= x || agents[a].y >= y) {
								++numChanges;
								bool success = performChange(REMOVE_AGENT, 0, 0, a--, 1);
								assert(success);
							}
						}

						roomWidth = x;
						roomHeight = y;

						// set the correct group sizes
						assert(undoTop - numChanges >= 0);
						if (numChanges != 0) {
							change.groupSize = 1 + numChanges;
							changeStack[undoTop - numChanges].groupSize =
								change.groupSize;
						}
					}
					else {
						// fill new space with FLOORs
						for (int cx = 0; cx < x; ++cx) {
							for (int cy = 0; cy < y; ++cy) {
								if (cx >= roomWidth || cy >= roomHeight) {
									room[cx][cy] = FLOOR;
								}
							}
						}
						roomWidth = x;
						roomHeight = y;
					}
				}
			} break;
			default: break;
		}

		if (commit) {
			// we actually changed something, commit the change to the undo stack
			int idx = undoTop;
			assert(idx <= MAX_UNDO);

			// we need to make sure not to overflow the change stack
			// if we ever make more changes than we can handle just
			// remove the last group of changes to make space
			// it might take a very long time if the buffer is big
			// but its the best we can do while keeping it simple
			if (idx == MAX_UNDO) {
				// remove last change so we can fit the new change
				int shiftSize = changeStack[0].groupSize;
				memmove(changeStack, &changeStack[shiftSize],
					(MAX_UNDO - shiftSize) * sizeof(changeStack[0]));

				idx -= shiftSize;
				undoTop -= shiftSize;
				redoTop -= shiftSize;

				assert(idx >= 0 && idx < MAX_UNDO);
				assert(undoTop >= 0 && undoTop < MAX_UNDO);
				assert(redoTop >= 0 && redoTop < MAX_UNDO);
			}

			++undoTop;
			redoTop = undoTop;
			changeStack[idx] = change;
		}

		return commit;
	}
	else
	{
		return FALSE;
	}
}

// undo last changes made by performChange
void undo() {
	if (undoTop > 0 && uiState == EDITING) {
		int groupSize = changeStack[undoTop - 1].groupSize;
		assert(groupSize > 0);

		// undo every change in the change group
		for (int c = 0; c < groupSize; ++c) {
			assert(undoTop > 0);
			--undoTop;
			changeinfo change = changeStack[undoTop];

			switch (change.change) {
				case REPLACE_CELL: {
					int x = change.replaceCell.x;
					int y = change.replaceCell.y;
					char newCell = change.replaceCell.newCell;
					char oldCell = change.replaceCell.oldCell;
					assert(isInRoom(x, y));
					assert(newCell != oldCell);
					room[x][y] = oldCell;
				} break;
				case INSERT_AGENT: {
					int x = change.insertAgent.x;
					int y = change.insertAgent.y;
					int a = change.insertAgent.agentIndex;
					int h = change.insertAgent.agentHealth;
					assert(isInRoom(x, y));
					assert(a >= 0 && a < numAgents);
					assert(h >= 0 && h <= MAX_HEALTH);
					removeAgent(a);
				} break;
				case REMOVE_AGENT: {
					int x = change.removeAgent.x;
					int y = change.removeAgent.y;
					int a = change.removeAgent.agentIndex;
					int h = change.removeAgent.agentHealth;
					assert(isInRoom(x, y));
					assert(a >= 0 && a <= numAgents);
					assert(h >= 0 && h <= MAX_HEALTH);
					insertAgent(a, x, y);
					agents[a].health = h;
				} break;
				case RESIZE_ROOM: {
					int newW = change.resizeRoom.newWidth;
					int newH = change.resizeRoom.newHeight;
					int oldW = change.resizeRoom.oldWidth;
					int oldH = change.resizeRoom.oldHeight;
					assert(newW != oldW || newH != oldH);
					assert(newW > 0 && newW <= MAX_ROOM_SIZE);
					assert(oldW > 0 && oldW <= MAX_ROOM_SIZE);
					assert(newH > 0 && newH <= MAX_ROOM_SIZE);
					assert(oldH > 0 && oldH <= MAX_ROOM_SIZE);
					roomWidth  = oldW;
					roomHeight = oldH;
				} break;
				default: assert(FALSE); break;
			}
		}
	}
}

// undo the undo
void redo() {
	if (undoTop < redoTop && uiState == EDITING) {
		int groupSize = changeStack[undoTop].groupSize;
		assert(groupSize > 0);

		// redo every change in the group
		for (int c = 0; c < groupSize; ++c) {
			assert(undoTop < redoTop);
			changeinfo change = changeStack[undoTop];
			++undoTop;

			switch (change.change) {
				case REPLACE_CELL: {
					int x = change.replaceCell.x;
					int y = change.replaceCell.y;
					char newCell = change.replaceCell.newCell;
					char oldCell = change.replaceCell.oldCell;
					assert(isInRoom(x, y));
					assert(newCell != oldCell);
					room[x][y] = newCell;
				} break;
				case INSERT_AGENT: {
					int x = change.insertAgent.x;
					int y = change.insertAgent.y;
					int a = change.insertAgent.agentIndex;
					int h = change.removeAgent.agentHealth;
					assert(isInRoom(x, y));
					assert(a >= 0 && a <= numAgents);
					assert(h >= 0 && h <= MAX_HEALTH);
					insertAgent(a, x, y);
					agents[a].health = h;
				} break;
				case REMOVE_AGENT: {
					int x = change.insertAgent.x;
					int y = change.insertAgent.y;
					int a = change.insertAgent.agentIndex;
					int h = change.removeAgent.agentHealth;
					assert(isInRoom(x, y));
					assert(a >= 0 && a < numAgents);
					assert(h >= 0 && h <= MAX_HEALTH);
					removeAgent(a);
				} break;
				case RESIZE_ROOM: {
					int newW = change.resizeRoom.newWidth;
					int newH = change.resizeRoom.newHeight;
					int oldW = change.resizeRoom.oldWidth;
					int oldH = change.resizeRoom.oldHeight;
					assert(newW != oldW || newW != oldH);
					assert(newW > 0 && newW <= MAX_ROOM_SIZE);
					assert(oldW > 0 && oldW <= MAX_ROOM_SIZE);
					assert(newH > 0 && newH <= MAX_ROOM_SIZE);
					assert(oldH > 0 && oldH <= MAX_ROOM_SIZE);
					roomWidth  = newW;
					roomHeight = newH;
				} break;
				default: assert(FALSE); break;
			}
		}
	}
}

// convert float RGBA colot to byte RGBA
rgba fRGBA(double r, double g, double b, double a) {
	// make sure colors are in range
	r = clampf(r, 0, 1);
	g = clampf(g, 0, 1);
	b = clampf(b, 0, 1);
	a = clampf(a, 0, 1);

	if (grayscale) {
		// simple conversion from:
		// https://en.wikipedia.org/wiki/Grayscale
		r = g = b =
			0.2126 * r +
			0.7152 * g +
			0.0722 * b;
	}

	GLubyte bytes[4];
	// multiply by 255.5 because it
	// has more balanced rounding than 255.0
	bytes[0] = (GLubyte)(r * 255.5);
	bytes[1] = (GLubyte)(g * 255.5);
	bytes[2] = (GLubyte)(b * 255.5);
	bytes[3] = (GLubyte)(a * 255.5);
	return *(rgba *)bytes;
}

// convert float HSVA color to byte RGBA
rgba fHSVA(double h, double s, double v, double a) {
	// make sure inputs are in range
	h = clampf(h, 0, 1);
	s = clampf(s, 0, 1);
	v = clampf(v, 0, 1);
	a = clampf(a, 0, 1);

	// adapted from:
	// https://en.wikipedia.org/wiki/HSL_and_HSV
	int i = (int)(h * 6);
	double f = h * 6 - i;
	double p = v * (1 - s);
	double q = v * (1 - f * s);
	double t = v * (1 - (1 - f) * s);

	double r, g, b;
	switch (i % 6) {
		case 0:  r = v; g = t; b = p; break;
		case 1:  r = q; g = v; b = p; break;
		case 2:  r = p; g = v; b = t; break;
		case 3:  r = p; g = q; b = v; break;
		case 4:  r = t; g = p; b = v; break;
		default: r = v; g = p; b = q; break;
	}

	return fRGBA(r, g, b, a);
}

// get mouse position in cell coordinates
void mouseCellPos(int *x, int *y) {
	// transform (mouseX, mouseY) from screen to cell coords
	*x = (int)floor((mouseX - transX) / scale);
	*y = (int)floor((mouseY - transY) / scale);
}

// draw an embezzled rectangle like so:
//                          (x1,y1)
//     bbbbbbbbbbbbbbbbbbbbbbbbb
//     b                       b
//     b  eeeeeeeeeeeeeeeeeee  b
//     b  e                 e  b
//     b  e  ttttttttttttt  e  b
//     b  e                 e  b
//     b  eeeeeeeeeeeeeeeeeee  b
//     b                       b
//     bbbbbbbbbbbbbbbbbbbbbbbbb
// (x0,y0)
//
// b = bottom, e = edge, t = top
void drawBezel(double x0, double y0, double x1, double y1, double edgeProp, rgba topColor, rgba edgeColor, rgba bottomColor) {
	double ew = fmin(fabs(x1 - x0), fabs(y1 - y0)) * edgeProp;

	GLubyte *tc = (GLubyte *)&topColor;
	GLubyte *ec = (GLubyte *)&edgeColor;
	GLubyte *bc = (GLubyte *)&bottomColor;
	glBegin(GL_QUADS);

	// top
	glColor4ubv(tc);
	glVertex2d(x0 + ew, y0 + ew);
	glVertex2d(x1 - ew, y0 + ew);
	glVertex2d(x1 - ew, y1 - ew);
	glVertex2d(x0 + ew, y1 - ew);

	if (edgeProp != 0) {
		// left edge
		glColor4ubv(ec);
		glVertex2d(x0 + ew, y0 + ew);
		glVertex2d(x0 + ew, y1 - ew);
		glColor4ubv(bc);
		glVertex2d(x0, y1);
		glVertex2d(x0, y0);

		// right edge
		glColor4ubv(ec);
		glVertex2d(x1 - ew, y0 + ew);
		glVertex2d(x1 - ew, y1 - ew);
		glColor4ubv(bc);
		glVertex2d(x1, y1);
		glVertex2d(x1, y0);

		// top edge
		glColor4ubv(ec);
		glVertex2d(x0 + ew, y1 - ew);
		glVertex2d(x1 - ew, y1 - ew);
		glColor4ubv(bc);
		glVertex2d(x1, y1);
		glVertex2d(x0, y1);

		// bottom edge
		glColor4ubv(ec);
		glVertex2d(x0 + ew, y0 + ew);
		glVertex2d(x1 - ew, y0 + ew);
		glColor4ubv(bc);
		glVertex2d(x1, y0);
		glVertex2d(x0, y0);
	}

	glEnd();
}

// draw a elliptical pie centered at (x,y)
void drawPie(double x, double y, double radiusX, double radiusY, double startAngle, double endAngle, rgba color) {
	// we are approximating the elipse with triangles
	// so if we want to have a consistent level of detail
	// regardless of zoom we can make the number of triangles
	// dependent on the radius
	const double D = 4; // 1/detail

	double r = fmax(radiusX * scale, radiusY * scale);
	int numTriangles = (int)clampf(
		2 / (1 - 2 / pi * acos(D / (2 * r))),
		6, fmax(windowWidth, windowHeight));
	double angleStep = 2 * pi / numTriangles;

	glBegin(GL_TRIANGLE_FAN);

	glColor4ubv((GLubyte *)&color);
	glVertex2d(x, y);

	for (double angle = startAngle; angle < endAngle; angle += angleStep) {
		glVertex2d(
			x + radiusX * cos(angle),
			y + radiusY * sin(angle));
	}

	glVertex2d(
		x + radiusX * cos(endAngle),
		y + radiusY * sin(endAngle));

	glEnd();
}

// draw given cell at bottom-left = (x,y)
void drawCell(char c, double x, double y, double size, double opacity, void *address) {
	switch (c) {
		case FLOOR: {
			rgba topColor = fRGBA(.8, .8, .8, opacity);
			rgba botColor = fRGBA(.6, .6, .6, opacity);
			drawBezel(x, y, x + size, y + size, 0.04, topColor, topColor, botColor);
		} break;
		case WALL: {
			rgba topColor = fRGBA(.3, .3, .3, opacity);
			rgba botColor = fRGBA(.1, .1, .1, opacity);
			drawBezel(x, y, x + size, y + size, 0.2, topColor, topColor, botColor);
		} break;
		case EXIT: {
			// ground
			rgba topColor = fRGBA(.1, .7, .2, opacity);
			rgba botColor = fRGBA(.1, .4, .2, opacity);
			drawBezel(x, y, x + size, y + size, 0.05, topColor, topColor, botColor);

			// flag pole
			rgba black = fRGBA(0, 0, 0, opacity);
			rgba gray = fRGBA(0.5, 0.5, 0.5, opacity);
			double m = size / 18;
			drawBezel(x + 6 * m, y + 2 * m, x + 7 * m, y + size - 2 * m, 0.2, gray, black, black);

			// flag
			rgba red = fRGBA(1, 0, 0, opacity);
			glBegin(GL_TRIANGLES);
			glColor4ubv((GLubyte *)&red);
			glVertex2d(x + 7 * m, y + 15 * m);
			glVertex2d(x + 15 * m, y + 13 * m);
			glVertex2d(x + 7 * m, y + 11 * m);
			glEnd();
		} break;
		case GLASS: {
			drawCell(FLOOR, x, y, size, opacity, NULL);
			rgba topColor = fRGBA(0.1, 0.5, 1, 0.3);
			rgba botColor = fRGBA(0.1, 0.5, 1, 0.7);
			drawBezel(x, y, x + size, y + size, 0.2, topColor, topColor, botColor);
		} break;
		case SHARDS: {
			drawCell(FLOOR, x, y, size, opacity, NULL);

			rgba color = fRGBA(0.1, 0.5, 1, 0.4);
			glBegin(GL_TRIANGLES);
			glColor4ubv((GLubyte *)&color);

			rng rng = seedRNG((int)(x * MAX_ROOM_SIZE + y));

			double cw = size / 3;
			for (int i = 0; i < 4; ++i) {
				for (double sx = x; sx < x + size; sx += cw) {
					for (double sy = y; sy < y + size; sy += cw) {
						double v[3][2];
						v[0][0] = clampf(sx          + (cw / 2) * randf(&rng), x, x + size);
						v[0][1] = clampf(sy          + (cw / 2) * randf(&rng), y, y + size);
						v[1][0] = clampf(sx + cw     - (cw / 2) * randf(&rng), x, x + size);
						v[1][1] = clampf(sy          + (cw / 2) * randf(&rng), y, y + size);
						v[2][0] = clampf(sx + cw / 2 + (cw / 2) * randf(&rng), x, x + size);
						v[2][1] = clampf(sy + cw     - (cw / 2) * randf(&rng), y, y + size);

						glVertex2dv(v[0]);
						glVertex2dv(v[1]);
						glVertex2dv(v[2]);
					}
				}
				cw *= 0.75;
			}

			glEnd();
		} break;
		case DOOR:
		case OPEN_DOOR: {
			// we want the door to connect to neighboring cells
			// so that it doesnt look like a stupid chunky brick
			// in order to determine whether the door should connect
			// horizontally or vertically we count the number of
			// neighboring cells in both orientations and the
			// orientation with the most neighbors wins, in case
			// of a tie the door is simply not aligned
			int vNeighbors = 0;
			int hNeighbors = 0;

			// use address to figure out the indices
			// into the global room array
			char *ca = address;
			if (ca >= &room[0][0] && ca <= &room[MAX_ROOM_SIZE - 1][MAX_ROOM_SIZE - 1]) {
				// cell is actually a part of the room
				int offset = (int)(ca - &room[0][0]);
				int cx = offset / MAX_ROOM_SIZE;
				int cy = offset % MAX_ROOM_SIZE;

				// check all neighbor cells to see where the door should connect
				hNeighbors += (!isInRoom(cx - 1, cy) || room[cx - 1][cy] != FLOOR);
				hNeighbors += (!isInRoom(cx + 1, cy) || room[cx + 1][cy] != FLOOR);
				vNeighbors += (!isInRoom(cx, cy - 1) || room[cx][cy - 1] != FLOOR);
				vNeighbors += (!isInRoom(cx, cy + 1) || room[cx][cy + 1] != FLOOR);
			}

			rgba topColor = fRGBA(.3, .1, 0, opacity);
			rgba botColor = fRGBA(0.5, 0.3, 0, opacity);

			double cw = size / 2;
			double cx = x + cw;
			double cy = y + cw;

			drawCell(FLOOR, x, y, size, opacity, NULL);
			if (c == DOOR) {
				if (vNeighbors == hNeighbors) {
					drawBezel(x, y, x + size, y + size, 0.2, topColor, topColor, botColor);
				} else if (hNeighbors > vNeighbors) {
					drawBezel(x, cy - 0.35 * cw, x + size, cy + 0.35 * cw, 0.2, topColor, topColor, botColor);
					drawBezel(cx - 0.05 * cw, cy - 0.35 * cw, cx + 0.05 * cw, cy + 0.35 * cw,
						0.3, topColor, topColor, botColor);
				} else if (vNeighbors > hNeighbors) {
					drawBezel(cx - 0.35 * cw, y, cx + 0.35 * cw, y + size, 0.2, topColor, topColor, botColor);
					drawBezel(cx - 0.35 * cw, cy - 0.05 * cw, cx + 0.35 * cw, cy + 0.05 * cw,
						0.3, topColor, topColor, botColor);
				}
			} else if (c == OPEN_DOOR) {
				if (vNeighbors == hNeighbors) {
					drawBezel(x, y, x + 0.2 * size, y + 0.2 * size, 0.4, topColor, topColor, botColor);
					drawBezel(x + 0.8 * size, y, x + size, y + 0.2 * size, 0.4, topColor, topColor, botColor);
					drawBezel(x + 0.8 * size, y + 0.8 * size, x + size, y + size, 0.4, topColor, topColor, botColor);
					drawBezel(x, y + 0.8 * size, x + 0.2 * size, y + size, 0.4, topColor, topColor, botColor);
				} else if (hNeighbors > vNeighbors) {
					drawBezel(x, cy - 0.35 * cw, x + 0.1 * size, cy + 0.35 * cw, 0.4, topColor, topColor, botColor);
					drawBezel(x + 0.9 * size, cy - 0.35 * cw, x + size, cy + 0.35 * cw, 0.4, topColor, topColor, botColor);
				} else if (vNeighbors > hNeighbors) {
					drawBezel(cx - 0.35 * cw, y, cx + 0.35 * cw, y + 0.1 * size, 0.2, topColor, topColor, botColor);
					drawBezel(cx - 0.35 * cw, y + 0.9 * size, cx + 0.35 * cw, y + size, 0.2, topColor, topColor, botColor);
				}
			}
		} break;
		case BANDAGE: {
			drawCell(FLOOR, x, y, size, opacity, NULL);
			double cw = size / 2;
			double cx = x + cw;
			double cy = y + cw;
			double bw = size / 8;

			// draw cross
			rgba red = fRGBA(1, 0, 0, opacity);
			drawBezel(cx - bw, y + bw, cx + bw, y + 2 * cw - bw, 0, red, red, red);
			drawBezel(x + bw, cy - bw, x + 2 * cw - bw, cy + bw, 0, red, red, red);
		} break;
		case AGENT: {
			x += size / 2;
			y += size / 2;

			// draw black border
			rgba black = fRGBA(0, 0, 0, 1);
			double r0 = 0.35 * size;
			double r1 = 0.30 * size;
			drawPie(x, y, r0, r0, 0, 2 * pi, black);

			// draw the inside as a pie
			// representing the agents health
			agent *aa = address;
			double h = 0;
			double p = 2 * pi;
			if (aa >= &agents[0] && aa < &agents[numAgents]) {
				h = (aa - &agents[0]) / (double)numAgents;
				p = 2 * pi * aa->health / MAX_HEALTH;
			}

			rgba colorF = fHSVA(h, 1, 1, 1.0);
			rgba colorE = fHSVA(h, 1, 1, 0.5);
			drawPie(x, y, r1, r1, 0, p, colorF);
			drawPie(x, y, r1, r1, p, 2 * pi, colorE);

			double gw = size * 0.080;
			double gh = size * 0.045;
			double lx = x - size / 8;
			double ly = y - size / 10;
			double rx = x + size / 8;
			double ry = ly;

			drawBezel(lx - gw, ly - gh, lx + gw, ly + gh, 0, black, black, black);
			drawBezel(rx - gw, ry - gh, rx + gw, ry + gh, 0, black, black, black);
			drawBezel(lx, ly - 0.01 * size, rx, ly + 0.01 * size, 0, black, black, black);
		} break;
		default: {
			// this cell isnt being rendered properly
			// color it magenta so its easy to notice
			rgba color = fRGBA(1, 0, 1, 1);
			drawBezel(x, y, x + size, y + size, 0, color, 0, 0);
		} break;
	}
}

// draw an overlay that represents the Q-values
void drawCellQValues(int x, int y) {
	assert(visualizeQTable != 0);

	// since getQEntry needs a valid agent index, we use
	// a "fake" agent here which we move to the cell location
	// and then get the Q-entry .. since we have to use such
	// stupid workarounds it probably means getQEntry could be
	// designed a bit better..
	//@TODO fix above?
	int fakeId = MAX_AGENTS - 1; // use the last agent
	agent backup = agents[fakeId];
	agents[fakeId].health = visualizeQTable;
	agents[fakeId].x = x;
	agents[fakeId].y = y;

	double *qA, *qB;
	getQEntry(fakeId, &qA, &qB);

	// restore the old agent just in case
	agents[fakeId] = backup;

	// get colors for each action in this cell for the current state
	// red colors are used for negative values and green for positive
	rgba actionColors[5];
	for (action a = STAY; a <= UP; ++a) {
		double q = useDoubleQ ? (qA[a] + qB[a]) / 2 : qA[a];
		double red   = q < 0;
		double green = q > 0;
		double opacity = 0;

		// we scale the color with sqrt to make the
		// transition between 0 and visible a bit faster
		if (q > 0) {
			opacity = 0.5 * sqrt(q / escapeReward);
		} else if (q < 0) {
			opacity = 0.5 * sqrt(q / deathPunishment);
		}

		actionColors[a] = fRGBA(red, green, 0, opacity);
	}

	// the best action is highlighted in blue
	action bestAction = getBestAction(qA, qB);
	rgba bestActionColor = fRGBA(0, 0, 1, 0.2);

	// now we need to actually draw the values in
	// sadly we cant really use drawBezel for this
	double x0 = x;
	double x1 = x + 1.0;
	double y0 = y;
	double y1 = y + 1.0;
	double ew = fmin(fabs(x1 - x0), fabs(y1 - y0)) * 0.3;

	GLubyte *sc = (GLubyte *)&actionColors[STAY];
	GLubyte *lc = (GLubyte *)&actionColors[LEFT];
	GLubyte *rc = (GLubyte *)&actionColors[RIGHT];
	GLubyte *dc = (GLubyte *)&actionColors[DOWN];
	GLubyte *uc = (GLubyte *)&actionColors[UP];
	GLubyte *bc = (GLubyte *)&bestActionColor;
	glBegin(GL_QUADS);

	// top
	glColor4ubv(sc);
	glVertex2d(x0 + ew, y0 + ew);
	glVertex2d(x1 - ew, y0 + ew);
	glVertex2d(x1 - ew, y1 - ew);
	glVertex2d(x0 + ew, y1 - ew);
	if (STAY == bestAction) {
		glColor4ubv(bc);
		glVertex2d(x0 + ew, y0 + ew);
		glVertex2d(x1 - ew, y0 + ew);
		glVertex2d(x1 - ew, y1 - ew);
		glVertex2d(x0 + ew, y1 - ew);
	}

	// left edge
	glColor4ubv(lc);
	glVertex2d(x0 + ew, y0 + ew);
	glVertex2d(x0 + ew, y1 - ew);
	glVertex2d(x0, y1);
	glVertex2d(x0, y0);
	if (LEFT == bestAction) {
		glColor4ubv(bc);
		glVertex2d(x0 + ew, y0 + ew);
		glVertex2d(x0 + ew, y1 - ew);
		glVertex2d(x0, y1);
		glVertex2d(x0, y0);
	}

	// right edge
	glColor4ubv(rc);
	glVertex2d(x1 - ew, y0 + ew);
	glVertex2d(x1 - ew, y1 - ew);
	glVertex2d(x1, y1);
	glVertex2d(x1, y0);
	if (RIGHT == bestAction) {
		glColor4ubv(bc);
		glVertex2d(x1 - ew, y0 + ew);
		glVertex2d(x1 - ew, y1 - ew);
		glVertex2d(x1, y1);
		glVertex2d(x1, y0);
	}

	// top edge
	glColor4ubv(uc);
	glVertex2d(x0 + ew, y1 - ew);
	glVertex2d(x1 - ew, y1 - ew);
	glVertex2d(x1, y1);
	glVertex2d(x0, y1);
	if (UP == bestAction) {
		glColor4ubv(bc);
		glVertex2d(x0 + ew, y1 - ew);
		glVertex2d(x1 - ew, y1 - ew);
		glVertex2d(x1, y1);
		glVertex2d(x0, y1);
	}

	// bottom edge
	glColor4ubv(dc);
	glVertex2d(x0 + ew, y0 + ew);
	glVertex2d(x1 - ew, y0 + ew);
	glVertex2d(x1, y0);
	glVertex2d(x0, y0);
	if (DOWN == bestAction) {
		glColor4ubv(bc);
		glVertex2d(x0 + ew, y0 + ew);
		glVertex2d(x1 - ew, y0 + ew);
		glVertex2d(x1, y0);
		glVertex2d(x0, y0);
	}

	glEnd();

	rgba black = fRGBA(0, 0, 0, 1);

	// left edge
	glBegin(GL_LINE_LOOP);
	glColor4ubv((GLubyte *)&black);
	glVertex2d(x0, y0);
	glVertex2d(x0 + ew, y0 + ew);
	glVertex2d(x0 + ew, y1 - ew);
	glVertex2d(x0, y1);
	glEnd();

	// right edge
	glBegin(GL_LINE_LOOP);
	glColor4ubv((GLubyte *)&black);
	glVertex2d(x1 - ew, y0 + ew);
	glVertex2d(x1 - ew, y1 - ew);
	glVertex2d(x1, y1);
	glVertex2d(x1, y0);
	glEnd();

	// top edge
	glBegin(GL_LINE_LOOP);
	glColor4ubv((GLubyte *)&black);
	glVertex2d(x0 + ew, y1 - ew);
	glVertex2d(x1 - ew, y1 - ew);
	glVertex2d(x1, y1);
	glVertex2d(x0, y1);
	glEnd();

	// bottom edge
	glBegin(GL_LINE_LOOP);
	glColor4ubv((GLubyte *)&black);
	glVertex2d(x0 + ew, y0 + ew);
	glVertex2d(x1 - ew, y0 + ew);
	glVertex2d(x1, y0);
	glVertex2d(x0, y0);
	glEnd();
}

// draw the whole scene - the room, agents, and UI
void drawEverything() {
	// set up camera transform
	glLoadIdentity();
	glOrtho(0, windowWidth, 0, windowHeight, 0, 1000);
	glTranslated(transX, transY, 0);
	glScaled(scale, scale, 1);

	double tx = transX;
	double ty = transY;
	double  s = scale;

	grayscale = visualizeQTable != 0;

	// draw the whole room
	for (int x = 0; x < roomWidth; ++x) {
		for (int y = 0; y < roomHeight; ++y) {
			if (x * s + tx < windowWidth &&
				y * s + ty < windowHeight &&
				x * s + tx + s > 0 &&
				y * s + ty + s > 0)
			{
				// the cell is only drawn if its visible
				drawCell(room[x][y], x, y, 1, 1, &room[x][y]);
				if (visualizeQTable != 0) {
					grayscale = FALSE;
					drawCellQValues(x, y);
					grayscale = TRUE;
				}
			}
		}
	}

	// draw all the agents
	for (int a = 0; a < numAgents; ++a) {
		double x = agents[a].x;
		double y = agents[a].y;

		// if this agent is being dragged then render
		// it at the position of the mouse
		if (a == draggedAgent) {
			x = (mouseX - transX) / scale - 0.5;
			y = (mouseY - transY) / scale - 0.5;
		}

		if (x * s + tx < windowWidth &&
			y * s + ty < windowHeight &&
			x * s + tx + s > 0 &&
			y * s + ty + s > 0)
		{
			// the agent is only drawn if its visible
			drawCell(AGENT, x, y, 1, 1, &agents[a]);
		}
	}

	// highlight the cell with the mouse cursor
	int x, y;
	mouseCellPos(&x, &y);
	if (isInRoom(x, y)) {
		rgba color, borderColor;

		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
			color       = fRGBA(0, 1, 1, 0.2);
			borderColor = fRGBA(0, 1, 1, 0.3);
		} else {
			color       = fRGBA(0, 1, 1, 0.1);
			borderColor = fRGBA(0, 1, 1, 0.2);
		}

		if (selectedCell == AGENT) {
			drawPie(x + 0.5, y + 0.5, 0.42, 0.42, 0, 2 * pi, color);
			drawPie(x + 0.5, y + 0.5, 0.38, 0.38, 0, 2 * pi, color);
		} else {
			drawBezel(x, y, x + 1.0, y + 1.0, 0.05, color, borderColor, borderColor);
		}
	}

	glLoadIdentity();
	glOrtho(0, windowWidth, 0, windowHeight, 0, 1000);
	glTranslated(0, windowHeight - 100.0, 0);

	rgba black   = fRGBA(0.05, 0.05, 0.05, 0.5);
	drawBezel(10, 10, 90, 90, 0.1, black, black, black);

	if (uiState == EDITING && selectedCell != NONE) {
		// draw the selected cell in the top left
		drawCell(selectedCell, 20, 20, 60, 0.5, NULL);
	} else if (uiState == RUNNING) {
		// draw paused indicator in the top left
		rgba green = fRGBA(0, 1, 0, 0.5);
		glBegin(GL_TRIANGLES);
		glColor4ubv((GLubyte *)&green);
		glVertex2d(30, 20);
		glVertex2d(70, 50);
		glVertex2d(30, 80);
		glEnd();
	} else if (uiState == PAUSED) {
		// draw playing indicator in the top left
		rgba red     = fRGBA(1, 0, 0, 0.5);
		rgba darkRed = fRGBA(0.2, 0, 0, 0.5);
		drawBezel(20, 20, 45, 80, 0.2, red, red, darkRed);
		drawBezel(55, 20, 80, 80, 0.2, red, red, darkRed);
	}

	glLoadIdentity();
	grayscale = FALSE;
	if (visualizeQTable == 1) {
		// draw a red overlay for visualizing 1 health Q-values
		rgba red   = fRGBA(1, 0, 0, 0.4);
		rgba clear = fRGBA(1, 0, 0, 0);
		drawBezel(-1, -1, 1, 1, 0.02, clear, clear, red);
	} else if (visualizeQTable == 2) {
		// draw a green overlay for visualizing 2 health Q-values
		rgba green = fRGBA(0, 1, 0, 0.2);
		rgba clear = fRGBA(0, 1, 0, 0);
		drawBezel(-1, -1, 1, 1, 0.02, clear, clear, green);
	}
	grayscale = TRUE;
}

// select which cell to place when clicking
void selectCell(char newCell) {
	if (uiState == EDITING)  {
		if (newCell == SHARDS) {
			selectedCell = GLASS;
		} else if (newCell == OPEN_DOOR) {
			selectedCell = DOOR;
		} else if (newCell == FLOOR) {
			selectedCell = NONE;
		} else {
			selectedCell =
				newCell == NONE ? NONE :
				newCell == selectedCell ? NONE : newCell;
		}
	}
}

// switch from editing the room to running the simulation
void switchState(state newState) {
	if (newState != uiState)  {
		if (uiState == EDITING) {
			selectCell(NONE);
			currTurn = 0;
			totalReward = 0;
		}

		if (newState == EDITING) {
			memcpy(room, backupRoom, sizeof(room));
			memcpy(agents, backupAgents, sizeof(agents));
			currTurn = 0;
		}

		printf("%s\n",
			newState == EDITING ? "editing" :
			newState == RUNNING ? "running" :
			newState == PAUSED  ? "paused"  : "ERROR");

		uiState = newState;
	}
}

// center the camera on the room
void centerCamera() {
	double aspectRatio = (double)windowWidth / windowHeight;

	if (roomWidth > aspectRatio * roomHeight) {
		scale = (double)windowWidth / roomWidth;
		transY = (windowHeight - scale * roomHeight) / 2;
	} else {
		scale = (double)windowHeight / roomHeight;
		transY = 0;
	}

	transX = (windowWidth - scale * roomWidth) / 2;
}

// fires when mouse button is pressed/released
void onMouseClick(GLFWwindow *w, int button, int action, int mods) {
	if (action == GLFW_PRESS) {
		bool modIsDown =
			glfwGetKey(w, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
			glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
			glfwGetKey(w, GLFW_KEY_LEFT_ALT)      == GLFW_PRESS ||
			glfwGetKey(w, GLFW_KEY_RIGHT_ALT)     == GLFW_PRESS ||
			glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS ||
			glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS;

		if ((button == GLFW_MOUSE_BUTTON_LEFT && selectedCell == NONE) ||
		    (button == GLFW_MOUSE_BUTTON_LEFT && modIsDown))
		{
			// drag the screen
			dragging = TRUE;
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
			int x, y;
			mouseCellPos(&x, &y);

			// check if we clicked on an agent
			int a = agentAt(x, y);
			if (a >= 0) {
				performChange(REMOVE_AGENT, 0, 0, a, 1);
			} else {
				performChange(REPLACE_CELL, x, y, FLOOR, 1); // erase cell
			}
		} else if (button == GLFW_MOUSE_BUTTON_LEFT) {
			int x, y;
			mouseCellPos(&x, &y);

			if (isInRoom(x, y)) {
				int a = agentAt(x, y);
				if (selectedCell == AGENT) {
					// check if we should start dragging this agent
					if (a >= 0) {
						draggedAgent = a;
					} else if (!isPassable(room[x][y])) {
						printf("can't place Agent at (%d,%d) because %s is not passable\n",
							x, y, toString(room[x][y]));
					} else {
						if (!performChange(INSERT_AGENT, x, y, numAgents, 1)) {
							printf("can't place Agent at (%d,%d) because another Agent is in the way\n", x, y);
						}
					}
				} else if (selectedCell == GLASS && room[x][y] == GLASS) {
					performChange(REPLACE_CELL, x, y, SHARDS, 1);
				} else if (selectedCell == GLASS && room[x][y] == SHARDS) {
					performChange(REPLACE_CELL, x, y, GLASS, 1);
				} else if (selectedCell == DOOR && room[x][y] == DOOR) {
					performChange(REPLACE_CELL, x, y, OPEN_DOOR, 1);
				} else if (selectedCell == DOOR && room[x][y] == OPEN_DOOR) {
					performChange(REPLACE_CELL, x, y, DOOR, 1);
				} else if (agentAt(x, y) >= 0) {
					printf("can't place %s at (%d,%d) because and Agent is in the way\n",
						toString(selectedCell), x, y);
				} else {
					performChange(REPLACE_CELL, x, y, selectedCell, 1);
				}
			}
			else {
				printf("can't place %s outside of room\n", toString(selectedCell));
			}
		} else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
			int x, y;
			mouseCellPos(&x, &y);
			if (isInRoom(x, y)) {
				int a = agentAt(x, y);
				if (a >= 0) {
					selectCell(AGENT);
				} else {
					selectCell(room[x][y]);
				}
			}
			else {
				selectCell(NONE);
			}
		}
	} else if (action == GLFW_RELEASE) {
		dragging = FALSE;
		if (draggedAgent != NONE) {
			int x, y;
			mouseCellPos(&x, &y);
			if (x != agents[draggedAgent].x || y != agents[draggedAgent].y) {
				if (!isInRoom(x, y)) {
					printf("can't move Agent outside of room\n");
				} else if (!isPassable(room[x][y])) {
					printf("can't move Agent to (%d,%d) because %s is not passable\n",
						x, y, toString(room[x][y]));
				} else if (performChange(INSERT_AGENT, x, y, draggedAgent, 2)) {
					performChange(REMOVE_AGENT, 0, 0, draggedAgent + 1, 2);
				} else {
					printf("can't move Agent to (%d,%d) because another Agent is in the way\n", x, y);
				}
			}
			draggedAgent = NONE;
		}
	}
}

// fires when mouse moves in the window
void onMouseMove(GLFWwindow *w, double newX, double newY) {
	// (newX, newY) is in window coordinates
	// so Y = 0 is the top of the window
	newY = windowHeight - newY;

	int dx = (int)newX - mouseX;
	int dy = (int)newY - mouseY;
	mouseX = (int)newX;
	mouseY = (int)newY;

	if (dragging) {
		transX += dx;
		transY += dy;
	} else if (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		if (draggedAgent == NONE) {
			int x, y;
			mouseCellPos(&x, &y);
			if (selectedCell == AGENT) {
				performChange(INSERT_AGENT, x, y, numAgents, 1);
			} else {
				performChange(REPLACE_CELL, x, y, selectedCell, 1);
			}
		}
	} else if (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
		int x, y;
		mouseCellPos(&x, &y);
		int a = agentAt(x, y);
		if (a == NONE) {
			performChange(REPLACE_CELL, x, y, FLOOR, 1);
		} else {
			performChange(REMOVE_AGENT, 0, 0, a, 1);
		}
	}
}

// fires when scroll wheel is moved
void onScroll(GLFWwindow *w, double xoffset, double yoffset) {
	bool modIsDown =
		glfwGetKey(w, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
		glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
		glfwGetKey(w, GLFW_KEY_LEFT_ALT)      == GLFW_PRESS ||
		glfwGetKey(w, GLFW_KEY_RIGHT_ALT)     == GLFW_PRESS ||
		glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS ||
		glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS;

	if (modIsDown) {
		double oldScale = scale;

		if (yoffset > 0) {
			scale *= 1.05;
		} else if (yoffset < 0) {
			scale /= 1.05;
		}

		// we want to center on the mouse cursor
		// so translate everything so the pixel
		// under the mouse cursor stays put
		double sx = (mouseX - transX) / oldScale;
		double sy = (mouseY - transY) / oldScale;
		transX -= (scale - oldScale) * sx;
		transY -= (scale - oldScale) * sy;
	} else {
		const char wheel[] = {
			NONE,
			WALL,
			GLASS,
			DOOR,
			BANDAGE,
			EXIT,
			AGENT,
		};
		int numWheel = sizeof(wheel) / sizeof(wheel[0]);

		for (int i = 0; i < numWheel; ++i) {
			if (wheel[i] == selectedCell) {
				if (yoffset > 0) {
					selectCell(
						wheel[(i + 1) % numWheel]);
				} else if (yoffset < 0) {
					selectCell(
						wheel[(i + numWheel - 1) % numWheel]);
				}
				break;
			}
		}
	}
}

// fires when keyboard key is pressed or released
void onKeyPress(GLFWwindow *w, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, TRUE);
				break;
			case GLFW_KEY_H:
				printGUIHelp();
				break;
			case GLFW_KEY_C:
				centerCamera();
				break;
			case GLFW_KEY_PERIOD:
				switchState(PAUSED);
				simulateTurn();
				break;
			case GLFW_KEY_Z:
				if (mods & GLFW_MOD_CONTROL) {
					undo();
				} break;
			case GLFW_KEY_Y:
				if (mods & GLFW_MOD_CONTROL) {
					redo();
				} break;
			case GLFW_KEY_S:
				printf("the current room size is %dx%d.\n",
					roomWidth, roomHeight);
				break;
			case GLFW_KEY_EQUAL:
				if (mods != 0) {
					onScroll(window, 0, 1);
				} break;
			case GLFW_KEY_MINUS:
				if (mods != 0) {
					onScroll(window, 0, -1);
				} break;
			case GLFW_KEY_E:
				useEpsilon = !useEpsilon;
				if (useEpsilon) {
					printf("epsilon enabled\n");
				} else {
					printf("epsilon disabled\n");
				} break;
			case GLFW_KEY_V:
				visualizeQTable -= 1;
				if (visualizeQTable < 0) {
					visualizeQTable = MAX_HEALTH;
				}

				if (visualizeQTable != 0) {
					printf("Q-value visualizer %d health enabled\n",
						visualizeQTable);
				} else {
					printf("Q-value visualizer disabled\n");
				} break;
			case GLFW_KEY_F:
				fastMode = !fastMode;
				printf("%s\n", fastMode ? "fast mode" : "slow mode");
				if (fastMode) {
					glfwSwapInterval(0);
					switchState(RUNNING);
				} else {
					glfwSwapInterval(turnFreq < 30);
				} break;
			case GLFW_KEY_X: {
				glfwHideWindow(window);
				printf(">>> ");
				char input[256];
				if (fgets(input, sizeof(input), stdin)) {
					runCmd(input);
				}
				glfwShowWindow(window);
			} break;
			case GLFW_KEY_Q: {
				char command[64];
				sprintf(command, "setq %lg", optimism);
				runCmd(command);
				printf("Q-values set to %lg\n", optimism);
			} break;
			case GLFW_KEY_ENTER:
			case GLFW_KEY_SPACE: {
				bool space = key == GLFW_KEY_SPACE;
				bool enter = key == GLFW_KEY_ENTER;
				if (uiState == EDITING && (space || enter)) {
					switchState(RUNNING);
				}
				else if (uiState == RUNNING) {
					if (space) {
						switchState(PAUSED);
					} else if (enter) {
						switchState(EDITING);
					}
				}
				else if (uiState == PAUSED) {
					if (space) {
						switchState(RUNNING);
					} else if (enter) {
						switchState(EDITING);
					}
				}
			} break;
			case GLFW_KEY_LEFT:
				if (mods != 0) {
					transX += 16;
				} else {
					performChange(RESIZE_ROOM, roomWidth - 1, roomHeight, 0, 1);
				} break;
			case GLFW_KEY_RIGHT:
				if (mods != 0) {
					transX -= 16;
				} else {
					performChange(RESIZE_ROOM, roomWidth + 1, roomHeight, 0, 1);
				} break;
			case GLFW_KEY_UP:
				if (mods != 0) {
					transY -= 16;
				} else {
					performChange(RESIZE_ROOM, roomWidth, roomHeight + 1, 0, 1);
				} break;
			case GLFW_KEY_DOWN:
				if (mods != 0) {
					transY += 16;
				} else {
					performChange(RESIZE_ROOM, roomWidth, roomHeight - 1, 0, 1);
				} break;
			default: {
				if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
					if (uiState == EDITING)  {
						switch (key) {
							case GLFW_KEY_1: selectCell(WALL); break;
							case GLFW_KEY_2: selectCell(GLASS); break;
							case GLFW_KEY_3: selectCell(DOOR); break;
							case GLFW_KEY_4: selectCell(EXIT); break;
							case GLFW_KEY_5: selectCell(BANDAGE); break;
							default: selectCell(NONE); break;
						}
					} else if (!fastMode) {
						int val = key - GLFW_KEY_0;
						turnFreq = pow(2, val);
						printf("running %lg turns per second\n", turnFreq);
						bool vsync = turnFreq < 30;
						glfwSwapInterval(vsync);
					}
				}
			} break;
		}
	}
}

// fires while window is being resized
void onResize(GLFWwindow *w, int newWidth, int newHeight) {
	if (scale == 0) {
		scale = ((double)newHeight / roomHeight);
	} else {
		scale /= ((double)windowHeight / roomHeight);
		scale *= ((double)newHeight / roomHeight);
	}
	windowWidth  = newWidth;
	windowHeight = newHeight;
	glViewport(0, 0, newWidth, newHeight);
}

// open and run the UI
void runGUI() {
	// init GLFW
	int glfw = glfwInit();
	assert(glfw);

	// open window
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x MSAA
	window = glfwCreateWindow(1280, 720, "Escape", NULL, NULL);
	assert(window);

	// set up window events
	glfwSetMouseButtonCallback(window, onMouseClick);
	glfwSetCursorPosCallback(window, onMouseMove);
	glfwSetScrollCallback(window, onScroll);
	glfwSetWindowSizeCallback(window, onResize);
	glfwSetKeyCallback(window, onKeyPress);

	// set up globals
	double cx, cy;
	glfwGetCursorPos(window, &cx, &cy);
	glfwGetWindowSize(window, &windowWidth, &windowHeight);
	mouseX = (int)cx;
	mouseY = (int)cy;
	centerCamera();

	// load OpenGL functions and set up OpenGL
	glfwMakeContextCurrent(window);
	int glad = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	assert(glad);

	glEnable(GL_BLEND); // enable transparency
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glLineWidth(2);

	printf("\no========= Escape Room ==========o\n");
	printf(" press 'h' for a list of controls\n");
	printf("o================================o\n\n");

	glfwSwapInterval(1); // turn on vsync
	uint64_t t0 = glfwGetTimerValue();
	double timerPeriod = 1.0 / glfwGetTimerFrequency();

	// enter draw loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		uint64_t t1 = glfwGetTimerValue();
		double dt = timerPeriod * (t1 - t0);

		if (uiState == RUNNING) {
			int turnsPerFrame = fastMode ? 10000 : 1;
			if (dt >= 1 / turnFreq || fastMode) {
				t0 = t1;
				for (int s = 0; s < turnsPerFrame; ++s) {
					simulateTurn();
				}
			}
		} else {
			t0 = t1;
		}

		glClear(GL_COLOR_BUFFER_BIT);
		drawEverything();
		glfwSwapBuffers(window);
	}

	// we are going to save the room to disk now, so restore the backup
	if (currTurn > 0) {
		memcpy(room, backupRoom, sizeof(room));
		memcpy(agents, backupAgents, sizeof(agents));
	}

	// save room to room.txt
	printf("saving room.txt ... ");
	FILE *roomFile = fopen("room.txt", "wt");
	if (roomFile != NULL) {
		for (int y = roomHeight - 1; y >= 0; --y) {
			for (int x = 0; x < roomWidth; ++x) {
				if (agentAt(x, y) != NONE) {
					fputc(AGENT, roomFile);
				} else {
					fputc(room[x][y], roomFile);
				}
			}
			fputc('\n', roomFile);
		}
		fclose(roomFile);
		printf("done\n");
	} else {
		printf("couldn't write to file\n");
	}

	runCmd("quit");
}

#endif // NOGUI

int main() {
	runCmd("seed 42");
	runCmd("load room.txt");
#ifdef NOGUI
	runCLI();
#else
	runGUI();
#endif
}