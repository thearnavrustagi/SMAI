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

// Weights for different squares on the board
const int SQUARE_WEIGHTS[8][8] = {
    {500, -150, 30,  10,  10,  30, -150,  500},
    {-150, -250,  0,   0,   0,   0, -250, -150},
    {30,     0,  1,   2,   2,   1,    0,   30},
    {10,     0,  2,  16,  16,   2,    0,   10},
    {10,     0,  2,  16,  16,   2,    0,   10},
    {30,     0,  1,   2,   2,   1,    0,   30},
    {-150, -250,  0,   0,   0,   0, -250, -150},
    {500, -150, 30,  10,  10,  30, -150,  500}
};

// Check if a position is stable (cannot be flipped)
bool isStable(const char grid[8][8], int x, int y) {
    // Check if it's a corner
    if((x == 0 || x == 7) && (y == 0 || y == 7)) 
        return true;
    
    // Check horizontal stability
    bool horizontalStable = true;
    if(x > 0 && grid[x-1][y] == 'e') horizontalStable = false;
    if(x < 7 && grid[x+1][y] == 'e') horizontalStable = false;
    
    // Check vertical stability
    bool verticalStable = true;
    if(y > 0 && grid[x][y-1] == 'e') verticalStable = false;
    if(y < 7 && grid[x][y+1] == 'e') verticalStable = false;
    
    // Check diagonal stability
    bool diagonalStable = true;
    if(x > 0 && y > 0 && grid[x-1][y-1] == 'e') diagonalStable = false;
    if(x < 7 && y < 7 && grid[x+1][y+1] == 'e') diagonalStable = false;
    if(x > 0 && y < 7 && grid[x-1][y+1] == 'e') diagonalStable = false;
    if(x < 7 && y > 0 && grid[x+1][y-1] == 'e') diagonalStable = false;
    
    return horizontalStable && verticalStable && diagonalStable;
}

// Count mobility (number of valid moves)
int countMobility(const char grid[8][8], char player, char opponent) {
    int mobility = 0;
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            if(grid[i][j] == 'e') {
                // Check all 8 directions
                const int dx[] = {-1,-1,-1,0,0,1,1,1};
                const int dy[] = {-1,0,1,-1,1,-1,0,1};
                
                for(int dir = 0; dir < 8; dir++) {
                    int x = i + dx[dir];
                    int y = j + dy[dir];
                    bool foundOpponent = false;
                    
                    while(x >= 0 && x < 8 && y >= 0 && y < 8) {
                        if(grid[x][y] == opponent) foundOpponent = true;
                        else if(grid[x][y] == player && foundOpponent) {
                            mobility++;
                            break;
                        }
                        else break;
                        x += dx[dir];
                        y += dy[dir];
                    }
                }
            }
        }
    }
    return mobility;
}

// Enhanced evaluation function
double evaluatePosition(const OthelloBoard& board, Turn turn) {
    char grid[8][8];
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            Coin findTurn = board.get(i, j);
            if(findTurn == turn) grid[i][j] = 'y';
            else if(findTurn == other(turn)) grid[i][j] = 'm';
            else grid[i][j] = 'e';
        }
    }

    double score = 0;
    int myPieces = 0, oppPieces = 0;
    int myStablePieces = 0, oppStablePieces = 0;
    int totalPieces = 0;
    
    // Count pieces and calculate position value
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            if(grid[i][j] != 'e') totalPieces++;
            
            if(grid[i][j] == 'y') {
                myPieces++;
                if(isStable(grid, i, j)) {
                    myStablePieces++;
                    score += 30; // Bonus for stable pieces
                }
                score += SQUARE_WEIGHTS[i][j];
            }
            else if(grid[i][j] == 'm') {
                oppPieces++;
                if(isStable(grid, i, j)) {
                    oppStablePieces++;
                    score -= 30; // Penalty for opponent's stable pieces
                }
                score -= SQUARE_WEIGHTS[i][j];
            }
        }
    }

    // Mobility evaluation
    int myMobility = countMobility(grid, 'y', 'm');
    int oppMobility = countMobility(grid, 'm', 'y');
    
    // Early game: focus on mobility and position
    if(totalPieces < 20) {
        score += (myMobility - oppMobility) * 15;
        score += (myStablePieces - oppStablePieces) * 20;
    }
    // Mid game: balance between mobility, position and piece count
    else if(totalPieces < 45) {
        score += (myMobility - oppMobility) * 10;
        score += (myStablePieces - oppStablePieces) * 30;
        score += (myPieces - oppPieces) * 5;
    }
    // End game: focus on piece count and stable pieces
    else {
        score += (myStablePieces - oppStablePieces) * 40;
        score += (myPieces - oppPieces) * 20;
    }
    
    // Parity advantage (having the last move)
    if((64 - totalPieces) % 2 == 0) {
        score += 5;
    }

    return score;
}

// SSS* search with alpha-beta pruning (same as before)
double searchMove(OthelloBoard board, Move move, Turn turn, int depth, double alpha, double beta) {
    clock_t finish = clock();
    if(((double)(finish - start)/CLOCKS_PER_SEC) > 1.95) {
        return (depth & 1) ? -1e18 : 1e18;
    }

    if(depth == 6) {
        return evaluatePosition(board, myTurn);
    }

    board.makeMove(turn, move);
    turn = other(turn);
    list<Move> moves = board.getValidMoves(turn);

    if(moves.empty()) {
        return evaluatePosition(board, myTurn);
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

// Comparison function for move sorting (same as before)
bool compareMove(Move a, Move b) {
    OthelloBoard boardA = globalBoard, boardB = globalBoard;
    boardA.makeMove(myTurn, a);
    boardB.makeMove(myTurn, b);
    return evaluatePosition(boardA, myTurn) > evaluatePosition(boardB, myTurn);
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
