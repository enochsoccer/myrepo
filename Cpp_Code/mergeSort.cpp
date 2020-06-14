#include <vector>
#include <cmath> // floor
#include <iostream>

void merge(std::vector<int> &v, int l, int m, int r);
void mergeSort(std::vector<int> &v, int l, int r);
void printVect(std::vector<int> v);

int main()
{
   std::vector<int> v = {14,7,3,12,9,11,6,2}, vtemp = {14,7,3,12};
   mergeSort(v,0,v.size()-1);
   printVect(v);

   return 0;
}

void merge(std::vector<int> &v, int l, int m, int r)
{
   int firstArr_idx = l, secArr_idx = m+1;
   std::vector<int> vtemp; // for storing values in order
   int temp1, temp2; // values to compare from two arrays
   int f_i = 0, s_i = 0; // trackers for iterating through both arrays
   int firstArr_N = m-l+1, secArr_N = r-(m+1)+1;
   
   while(vtemp.size() < r-l+1) // while the ordered vector is not fully filled
   {
      temp1 = v[firstArr_idx + f_i];
      temp2 = v[secArr_idx + s_i];
      // Add smaller of two values and increment tracking index
      if(temp1 < temp2) {
      	vtemp.push_back(temp1);
      	++f_i;
      }
      else {
      	vtemp.push_back(temp2);
      	++s_i;
      }
      

      if(f_i >= firstArr_N) // if next index is out of bounds
         while(s_i < secArr_N) { // then add remaining elements from *other* array
      	    vtemp.push_back( v[secArr_idx + s_i] );
      	    ++s_i;
      	 }
      
      if(s_i >= secArr_N) // same as above
         while(f_i < firstArr_N) {
            vtemp.push_back( v[firstArr_idx + f_i] );
            ++f_i;
         }
   }
   
   // Rearranging vector
   for(int i = 0; i < vtemp.size(); ++i)
   	v[l+i] = vtemp[i];
}

void mergeSort(std::vector<int> &v, int l, int r)
{
   if(l < r)
   {
      int m = floor( (l+r)/2 );
      mergeSort(v,l,m);
      mergeSort(v,m+1,r);
      merge(v,l,m,r);
   }  
}

void printVect(std::vector<int> v)
{
   for(int i = 0; i < v.size(); ++i)
      std::cout << v[i] << " ";
   std::cout << '\n';
}
