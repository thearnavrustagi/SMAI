#include "Othello.h"
#include "OthelloBoard.h"
#include "OthelloPlayer.h"
#include <cstdlib>
#include <ctime>
#include <list>
#include <unordered_map>
#include <algorithm>

using namespace std;
using namespace Desdemona;

#define INF 1e18

Turn my;
clock_t start, finish;
OthelloBoard globalBoard;

bool canMove(char self, char opp, char *str) {
    if (str[0] != opp) return false;
    for (int ctr = 1; ctr < 8; ctr++) {
        if (str[ctr] == 'e') return false;
        if (str[ctr] == self) return true;
    }
    return false;
}

bool isLegalMove(char self, char opp, char grid[8][8], int startx, int starty) {
    if (grid[startx][starty] != 'e') return false;
    char str[10];
    int x, y, dx, dy, ctr;
    for (dy = -1; dy <= 1; dy++)
        for (dx = -1; dx <= 1; dx++) {
            // keep going if both velocities are zero
            if (!dy && !dx) continue;
            str[0] = '\0';
            for (ctr = 1; ctr < 8; ctr++) {
                x = startx + ctr * dx;
                y = starty + ctr * dy;
                if (x >= 0 && y >= 0 && x < 8 && y < 8) str[ctr - 1] = grid[x][y];
                else str[ctr - 1] = 0;
            }
            if (canMove(self, opp, str)) return true;
        }
    return false;
}

int numValidMoves(char self, char opp, char grid[8][8]) {
    int count = 0, i, j;
    for (i = 0; i < 8; i++) for (j = 0; j < 8; j++) if (isLegalMove(self, opp, grid, i, j)) count++;
    return count;
}

double othelloBoardEvaluator(char grid[8][8]) {
    // Implement your board evaluation function here
    return 0.0;
}

double testMyMove(OthelloBoard board, Move move, Turn turn, short level, double alpha, double beta) {
    finish = clock();
    if (((double)(finish - start) / CLOCKS_PER_SEC) > 1.95) {
        if (level & 1) return -INF;
        return INF;
    }
    if (level == 6) {
        char grid[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                Coin findTurn = board.get(i, j);
                if (findTurn == turn) grid[i][j] = 'y';
                else if (findTurn == other(turn)) grid[i][j] = 'm';
                else grid[i][j] = 'e';
            }
        }
        return othelloBoardEvaluator(grid);
    }
    board.makeMove(turn, move);
    turn = other(turn);
    list<Move> newMoves = board.getValidMoves(turn);
    list<Move>::iterator iter = newMoves.begin();
    double ret = -INF;
    if (level & 1) ret *= -1;
    if (!(newMoves.size())) return ret;
    for (; iter != newMoves.end(); iter++) {
        double curr = testMyMove(board, *iter, turn, level + 1, alpha, beta);
        if (level & 1) {
            ret = min(ret, curr);
            beta = min(beta, ret);
        } else {
            ret = max(ret, curr);
            alpha = max(alpha, ret);
        }
        if (beta <= alpha) break;
    }
    return ret;
}

double tester(OthelloBoard board,Turn turn) {
    char grid[8][8];
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
        Coin findTurn = board.get(i,j);
        if(findTurn == turn) grid[i][j] = 'm';
        else if(findTurn == other(turn)) grid[i][j] = 'y';
        else grid[i][j] = 'e';
        }
    }
    return othelloBoardEvaluator(grid);
}

bool compare(Move a, Move b) {
    OthelloBoard One = globalBoard,Two = globalBoard;
    One.makeMove(my,a);
    Two.makeMove(my,b);
    return tester(One,my)>tester(Two,my);
}

class MyBot : public OthelloPlayer {
public:
    MyBot(Turn turn);
    virtual Move play(const OthelloBoard& board);

private:
};

MyBot::MyBot(Turn turn)
    : OthelloPlayer(turn) {
}

Move MyBot::play(const OthelloBoard& board) {
    start = clock();
    list<Move> moves = board.getValidMoves(turn);
    my = turn;
    globalBoard = board;
    moves.sort(compare);
    list<Move>::iterator it = moves.begin();
    Move bestMove((*it).x, (*it).y);
    double retVal = -INF;
    double MAX = INF, MIN = -INF;
    OthelloBoard copyBoard = board;
    short level = 1;
    for (; it != moves.end(); it++) {
        double currValue = testMyMove(copyBoard, *it, turn, level, MIN, MAX);
        if (currValue > retVal) {
            retVal = currValue;
            bestMove = *it;
        }
        copyBoard = board;
    }
    return bestMove;
}

extern "C" {
    OthelloPlayer* createBot(Turn turn) {
        return new MyBot(turn);
    }

    void destroyBot(OthelloPlayer* bot) {
        delete bot;
    }
}