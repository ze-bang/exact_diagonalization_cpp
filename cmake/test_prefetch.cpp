// Test for __builtin_prefetch support
int main() {
    int x = 42;
    __builtin_prefetch(&x, 0, 3);
    return 0;
}
