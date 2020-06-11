#include <cstdlib> // rand, srand
#include <iostream> // cout, cin
#include <tuple>

using namespace std;

const int NROWS = 6;
const int NCOLS = 7;
const int NsROWS = 3; // s for subset
const int NsCOLS = 3;

template<class T, std::size_t rows, std::size_t cols>
void printMatr(T(&matrix)[rows][cols]);
void randMatrix(int (&matrix)[NROWS][NCOLS]);
void genSubMat(const int matrix[NROWS][NCOLS], int (&subMat)[NsROWS][NsCOLS]);
bool checkSubMat(const int i, const int j, const int haystack[NROWS][NCOLS], const int subMat[NsROWS][NsCOLS]);
tuple<int,int> needle(const int haystack[NROWS][NCOLS], const int subMat[NsROWS][NsCOLS]);

// -------------------------------------------------------------------

int main()
{
    int haystack[NROWS][NCOLS] = {0}, subMat[NsROWS][NsCOLS] = {0};
    randMatrix(haystack);
    printMatr(haystack);
    genSubMat(haystack,subMat);
    printMatr(subMat);
    auto [row,col] = needle(haystack,subMat);
    cout << "The needle is at (" << row << "," << col << ")." << endl;
    return 0;
}

// -------------------------------------------------------------------

void randMatrix(int (&matrix)[NROWS][NCOLS])
{
    // Creates a random integer matrix.
    cout << "rows = " << NROWS << ", ncols = " << NCOLS << endl;
    srand (time(NULL));
    for(int i = 0; i < NROWS; ++i)
        for(int j = 0; j < NCOLS; ++j)
            matrix[i][j] = rand() % 10; // random number: 0 to 9
    return;
}

template<class T, std::size_t rows, std::size_t cols>
void printMatr(T(&matrix)[rows][cols])
{
    // Prints the contents of the matrix.
    // Inefficient use of range-based for loop (C++20) but good practice
    // https://docs.microsoft.com/en-us/cpp/cpp/range-based-for-statement-cpp?view=vs-2019
    int count = 0;
    for(const auto &val : matrix)  // type inference by constant reference; no copies created, just observes in place
        for(const auto &v : val)
        {
            cout << v << " ";
            if(count%cols == (cols-1)) // 0 to (cols-1)
                cout << endl;
            count = ++count; // doesn't create copy of what was before increment
        }
    cout << endl;
    return;
    /*
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
        {
            cout << matrix[i][j] << " ";
            if(j == cols-1)
                cout << endl;
        }
    cout << endl;
    return;
    */
}

void genSubMat(const int matrix[NROWS][NCOLS], int (&subMat)[NsROWS][NsCOLS])
{
    // Generates a subset matrix from the given one.
    int n = rand() % (NROWS-NsROWS+1);
    int m = rand() % (NCOLS-NsCOLS+1);
    for(int i = 0; i < NsROWS; ++i)
        for(int j = 0; j < NsCOLS; ++j)
            subMat[i][j] = matrix[n+i][m+j];
    return;
}

bool checkSubMat(const int i, const int j, const int haystack[NROWS][NCOLS], const int subMat[NsROWS][NsCOLS])
{
    // Checks to see if the rest of the entire 'subMat' is found in 'haystack.'
    // Returns true or false.
    bool isEqual = true;
    for(int n = 0; n < NsROWS; ++n)
        for(int m = 0; m < NsCOLS; ++m)
            if(haystack[i+n][j+m] != subMat[n][m])
                return false;
    return true; // if it reaches here, then every element is equal
}

tuple<int,int> needle(const int haystack[NROWS][NCOLS], const int subMat[NsROWS][NsCOLS])
{
    // Returns the 2D-array indices, aka the needle in the haystack.
    int val = subMat[0][0];
    bool foundMat = false;
    for(int i = 0; i < NROWS; ++i)
        for(int j = 0; j < NCOLS; ++j)
        {
            if(haystack[i][j] == val)
                if(checkSubMat(i,j,haystack,subMat))  // if the needle is found
                    return {i,j};
        }
    return {-1,-1}; // if it reaches here, then the needle was not found... which is an error
}


/*
4 6 7 9 5 8 6
3 2 0 2 0 9 2
5 6 1 4 1 0 1
8 0 3 2 0 4 0
8 1 1 2 7 0 3
4 8 9 7 3 1 9

0 2 0
1 4 1
3 2 0
*/
