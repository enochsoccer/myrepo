#include <iostream>


void swap(int *P, int *Q);
void swap(int &A, int &B);

int main()
{
  int *a = NULL, *b = NULL;
  int c = 5, d = 6;
  a = &c; // 5
  b = &d; // 6
  swap(a,b); // by pointers
  std::cout << "a: " << *a << std::endl;
  std::cout << "b: " << *b << std::endl;
  
  int A = 5, B = 6;
  swap(A,B); // by reference
  std::cout << "A = " << A << ", B = " << B << std::endl;
  swap(&A,&B); // by pointers
  std::cout << "A = " << A << ", B = " << B << std::endl;
}

void swap(int *P, int *Q)
{
   /* Good reference: https://stackoverflow.com/questions/8403447/swapping-pointers-in-c-char-int
   This only swaps the private copies of our pointers that are passed as arguments into the 
     function.
   
   int *temp = P;
   P = Q;
   Q = temp;
   
   This is similar to below, where only private copies are switched, and not the actual variables
     passed into the function.
     
   void swap(int a, int b)
   {
     int temp = a;
     a = b;
     b = temp;
   }
   */
  // Swapping with pointers.
   int temp = *P;
   *P = *Q;
   *Q = temp;
}

void swap(int &A, int &B)
{
   // Swapping by reference.
   int temp = A;
   A = B;
   B = temp;
}










