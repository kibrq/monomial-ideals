from collections import defaultdict


def sage_linearity_test(monomials):

    import sage.all as sage

    symbols = defaultdict(lambda: len(symbols))
    monomials = [[symbols[x] for x in m] for m in monomials]
    
    R = sage.PolynomialRing(sage.GF(101), len(symbols), 'x')
    x = R.gens()
    
    monomials = [sage.prod([x[i] for i in m]) for m in monomials]
    
    for gen in sage.Ideal(monomials).syzygy_module():
        if not set(map(lambda x: x.degree(), gen)) <= {-1, 1}:
            return False
    return True


def M2_linearity_test(monomials, base_field = "ZZ/101"):
    from sage.all import macaulay2

    if len(monomials) <= 1:
        return True

    symbols = defaultdict(lambda: f"x_{len(symbols)}")
    monomials = ["*".join([symbols[x] for x in monomial]) for monomial in monomials]

    M2_program = """
    isLinear = I -> (
        d := (degree I_*_0)_0;
    {{d+1}} == max degrees source syz gens I
    );
    
    {base_field}[{symbols}];
    
    isLinear(ideal({monomials}))
    """.format(
        base_field = base_field,
        symbols = ','.join(symbols.values()),
        monomials = ','.join(monomials),
    )

    result = macaulay2(M2_program)
    return result.external_string() in ['true']


def sympy_linearity_test(monomials, base_field = None):
    import sympy as sp

    x = sp.IndexedBase("x")
    symbols = defaultdict(lambda: x[len(symbols) + 1])
    monomials = [sp.prod([symbols[x] for x in m]) for m in monomials]

    base_field = base_field or sp.GF(101)

    ring = base_field.old_poly_ring(*symbols.values())
    module = ring.free_module(1).submodule(*[[m] for m in monomials]).syzygy_module()

    linear_syzgens = [gen for gen in module.gens if '*' not in str(gen)]
    linear_submod = module.submodule(*linear_syzgens)
    return linear_submod == module


def linearity_test(monomials, backend="M2", **kwargs):
    backend = backend.lower()
    # if backend.lower() in ["sage"]:
    #     return sage_linearity_test(monomials, **kwargs)
    if backend in ['m2', 'macaulay2']:
        return M2_linearity_test(monomials, **kwargs)
    if backend in ['sympy', 'sp', 'python']:
        return sympy_linearity_test(monomials, **kwargs)
    raise ValueError(f"Unknown {backend=}")
