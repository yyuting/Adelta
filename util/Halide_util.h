#include "Halide.h"
#include <stdio.h>

using namespace Halide;

Expr sign(Expr node) {
    return select(node == 0, 0.f, select(node > 0, 1.f, -1.f));
}

Expr cast2f(Expr node) {
    return Halide::cast<float> (node);
}

Expr cast2b(Expr node) {
    return Halide::cast<bool> (node);
}

std::vector<Expr> select(Expr cond, std::vector<Expr> left, std::vector<Expr> right) {

    std::vector<Expr> ans;
    
    if (left.size() != right.size()) {
        printf("Error! size of left and right must be the same!\n");
        throw;
    }
    
    for (int i = 0; i < left.size(); i++) {
        ans.push_back(select(cond, left[i], right[i]));
    }
    
    return ans;
}

Expr dot(std::vector<Expr> a, std::vector<Expr> b, int len=3) {
        
    if (a.size() < len || b.size() < len) {
        printf("Error! In dot product, either of the input vector has insufficient length!\n");
        throw;
    }

    Expr ans = 0.f;
    for (int i = 0; i < len; i++) {
        ans += a[i] * b[i];
    }

    return ans;
}

std::vector<Expr> sub3(std::vector<Expr> a, std::vector<Expr> b) {
    return {a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2]};
}

Expr binary2float(std::vector<Expr> L, int base_idx=0) {

    Expr value = 0.f;
    for (int idx = 0; idx < L.size(); idx++) {
        int scale = pow(2, idx + base_idx);
        value += cast2f(L[idx]) * scale;
    }
    
    return value;
}

// Some randomly-generated integers.
#define C0 576942909
#define C1 1121052041
#define C2 1040796640

// Permute a 32-bit unsigned integer using a fixed psuedorandom
// permutation.
Expr rng32(const Expr &x) {
    if (!(x.type() == UInt(32))) {
        printf("rng32 fails because input type is not UInt32\n");
    }

    // A polynomial P with coefficients C0 .. CN induces a permutation
    // modulo 2^d iff:
    // 1) P(0) != P(1) modulo 2
    // 2) sum(i * Ci) is odd

    // (See http://en.wikipedia.org/wiki/Permutation_polynomial#Rings_Z.2FpkZ)

    // For a quadratic, this is only satisfied by:
    // C0 anything
    // C1 odd
    // C2 even

    // The coefficients defined above were chosen to satisfy this
    // property.

    // It's pretty random, but note that the quadratic term disappears
    // if inputs are the multiples of 2^16, and so you get a linear
    // sequence.  However, *that* linear sequence probably varies in
    // the low bits, so if you run in through the permutation again,
    // you should break it up. All actual use of this runs it through
    // multiple times in order to combine several inputs, so it should
    // be ok. The other flaw is it's a permutation, so you get no
    // collisions. Birthday paradox be damned.

    // However, it's exceedingly cheap to compute, as it only uses
    // vectorizable int32 muls and adds, and the resulting numbers:
    // - Have the correct moments for a uniform distribution
    // - Have no serial correlations in any of the bits
    // - Have a completely flat power spectrum
    // - Have no visible patterns

    // So I declare this good enough for image processing.

    // If it's just a const (which it often is), save the simplifier some work:
    if (const uint64_t *i = as_const_uint(x)) {
        return Halide::Internal::make_const(UInt(32), ((C2 * (*i)) + C1) * (*i) + C0);
    }

    return (((C2 * x) + C1) * x) + C0;
}

Expr our_random_int(const std::vector<Expr> &e) {
    if (e.size() == 0) {
        printf("random_int fails because no input argument is available.\n");
    }
    if (e[0].type() != Int(32) && e[0].type() != UInt(32)) {
        printf("random_int fails because first argument is not int32 or uint32\n");
    }
    // Permute the first term
    Expr result = rng32(cast(UInt(32), e[0]));
    for (size_t i = 1; i < e.size(); i++) {
        if (e[i].type() != Int(32) && e[i].type() != UInt(32)) {
            printf("random_int fails because %dth argument is not int32 or uint32\n", i);
        }
        // Add in the next term and permute again
        std::string name = Halide::Internal::unique_name('R');
        // If it's a const, save the simplifier some work
        const uint64_t *ir = as_const_uint(result);
        const uint64_t *ie = as_const_uint(e[i]);
        if (ir && ie) {
            result = rng32(Halide::Internal::make_const(UInt(32), (*ir) + (*ie)));
        } else {
            result = rng32(result + Halide::cast<uint32_t>(e[i]));
        }
    }
    return result;
}

Expr our_random_float(const std::vector<Expr> &e) {
    Expr result = our_random_int(e);
    // Set the exponent to one, and fill the mantissa with 23 random bits.
    result = (127 << 23) | (cast<uint32_t>(result) >> 9);
    // The clamp is purely for the benefit of bounds inference.
    return clamp(reinterpret(Float(32), result) - 1.0f, 0.0f, 1.0f);
}