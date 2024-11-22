#include "Othello.h"
#include "OthelloBoard.h"
#include "OthelloPlayer.h"
#include <cstdlib>
#include <ctime>
#include <list>
#include <algorithm>

using namespace std;
using namespace Desdemona;

clock_t start;
Turn myTurn;
OthelloBoard globalBoard;

// Integrated weighted square values from first implementation
const int SQUARE_WEIGHTS[8][8] = {
    { 20, -3, 11, 8, 8, 11, -3, 20 },
    { -3, -7, -4, 1, 1, -4, -7, -3 },
    { 11, -4, 2, 2, 2, 2, -4, 11 },
    { 8, 1, 2, -3, -3, 2, 1, 8 },
    { 8, 1, 2, -3, -3, 2, 1, 8 },
    { 11, -4, 2, 2, 2, 2, -4, 11 },
    { -3, -7, -4, 1, 1, -4, -7, -3 },
    { 20, -3, 11, 8, 8, 11, -3, 20 }
};

// Mobility and edge calculation functions
int moveCalc(char self, char opp, char grid[8][8]) {
    int count = 0;
    for (int i = 0; i < 8; i++) 
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] != 'e') continue;
            
            const int dx[] = {-1,-1,-1,0,0,1,1,1};
            const int dy[] = {-1,0,1,-1,1,-1,0,1};
            
            for (int dir = 0; dir < 8; dir++) {
                int x = i + dx[dir], y = j + dy[dir];
                bool foundOpp = false;
                
                while (x >= 0 && x < 8 && y >= 0 && y < 8) {
                    if (grid[x][y] == opp) foundOpp = true;
                    else if (grid[x][y] == self && foundOpp) {
                        count++;
                        break;
                    }
                    else break;
                    x += dx[dir];
                    y += dy[dir];
                }
            }
        }
    return count;
}

// Enhanced board evaluation function
double evaluateBoard(const OthelloBoard& board, Turn turn) {
    char grid[8][8];
    char my_color = 'y', opp_color = 'm';
    
    // Convert board to grid
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Coin findTurn = board.get(i, j);
            if (findTurn == turn) grid[i][j] = my_color;
            else if (findTurn == other(turn)) grid[i][j] = opp_color;
            else grid[i][j] = 'e';
        }
    }

    int myTiles = 0, oppTiles = 0;
    int myFrontTiles = 0, oppFrontTiles = 0;
    double p = 0.0, c = 0.0, l = 0.0, m = 0.0, f = 0.0, d = 0.0;

    const int X1[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int Y1[] = {0, 1, 1, 1, 0, -1, -1, -1};

    // Piece difference, frontier disks, and disk squares
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] == my_color) {
                d += SQUARE_WEIGHTS[i][j];
                myTiles++;
            } 
            else if (grid[i][j] == opp_color) {
                d -= SQUARE_WEIGHTS[i][j];
                oppTiles++;
            }
            
            // Frontier tiles calculation
            if (grid[i][j] != 'e') {
                for (int k = 0; k < 8; k++) {
                    int x = i + X1[k], y = j + Y1[k];
                    if (x >= 0 && x < 8 && y >= 0 && y < 8 && grid[x][y] == 'e') {
                        if (grid[i][j] == my_color) myFrontTiles++;
                        else oppFrontTiles++;
                        break;
                    }
                }
            }
        }
    }

    // Piece difference score
    if (myTiles > oppTiles) p = (100.0 * myTiles) / (myTiles + oppTiles);
    else if (myTiles < oppTiles) p = -(100.0 * oppTiles) / (myTiles + oppTiles);

    // Frontier tiles score
    if (myFrontTiles > oppFrontTiles) 
        f = -(100.0 * myFrontTiles) / (myFrontTiles + oppFrontTiles);
    else if (myFrontTiles < oppFrontTiles) 
        f = (100.0 * oppFrontTiles) / (myFrontTiles + oppFrontTiles);

    // Corner occupancy
    int myCorners = 0, oppCorners = 0;
    int cornerPositions[][2] = {{0, 0}, {0, 7}, {7, 0}, {7, 7}};
    for (auto& pos : cornerPositions) {
        if (grid[pos[0]][pos[1]] == my_color) myCorners++;
        else if (grid[pos[0]][pos[1]] == opp_color) oppCorners++;
    }
    c = 25 * (myCorners - oppCorners);

    // Mobility
    int myMobility = moveCalc(my_color, opp_color, grid);
    int oppMobility = moveCalc(opp_color, my_color, grid);
    if (myMobility > oppMobility) 
        m = (100.0 * myMobility) / (myMobility + oppMobility);
    else if (myMobility < oppMobility) 
        m = -(100.0 * oppMobility) / (myMobility + oppMobility);

    // Final weighted score
    double score = (100 * p) + (490 * c) + (380 * l) + (86 * m) + (78 * f) + (16 * d);
    return score;
}

// SSS* search with alpha-beta pruning
double searchMove(OthelloBoard board, Move move, Turn turn, int depth, double alpha, double beta) {
    clock_t finish = clock();
    if(((double)(finish - start)/CLOCKS_PER_SEC) > 1.95) {
        return (depth & 1) ? -1e18 : 1e18;
    }

    if(depth == 6) {
        return evaluateBoard(board, myTurn);
    }

    board.makeMove(turn, move);
    turn = other(turn);
    list<Move> moves = board.getValidMoves(turn);

    if(moves.empty()) {
        return evaluateBoard(board, myTurn);
    }

    double value = (depth & 1) ? 1e18 : -1e18;
    
    for(Move nextMove : moves) {
        double curr = searchMove(board, nextMove, turn, depth + 1, alpha, beta);
        
        if(depth & 1) {
            value = min(value, curr);
            beta = min(beta, value);
        } else {
            value = max(value, curr);
            alpha = max(alpha, value);
        }
        
        if(beta <= alpha) break;
    }

    return value;
}

// Comparison function for move sorting
bool compareMove(Move a, Move b) {
    OthelloBoard boardA = globalBoard, boardB = globalBoard;
    boardA.makeMove(myTurn, a);
    boardB.makeMove(myTurn, b);
    return evaluateBoard(boardA, myTurn) > evaluateBoard(boardB, myTurn);
}

class MyBot : public OthelloPlayer {
public:
    MyBot(Turn turn) : OthelloPlayer(turn) {}
    
    virtual Move play(const OthelloBoard& board) {
        start = clock();
        myTurn = turn;
        globalBoard = board;
        
        list<Move> moves = board.getValidMoves(turn);
        if(moves.empty()) return Move::pass();
        
        moves.sort(compareMove);
        Move bestMove = moves.front();
        double bestValue = -1e18;
        
        OthelloBoard tempBoard;
        for(Move move : moves) {
            tempBoard = board;
            double value = searchMove(tempBoard, move, turn, 1, -1e18, 1e18);
            if(value > bestValue) {
                bestValue = value;
                bestMove = move;
            }
        }
        
        return bestMove;
    }
};

extern "C" {
    OthelloPlayer* createBot(Turn turn) {
        return new MyBot(turn);
    }
}
