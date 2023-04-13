#include <iostream>
void test(int& ref);
int main() {
    int* pointer = new int(10);               
    std::cout << "before " << *pointer << std::endl ;
    test(*pointer);
    std::cout << "after  " << *pointer << std::endl;
    delete pointer;
}

void test(int& ref) {    
    ref = 20;
}