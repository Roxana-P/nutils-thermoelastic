from nutils import function, export, mesh, solver, testing, cli
from nutils.expression_v2 import Namespace
import treelog
import numpy as np
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable


def get_cylinder(inner_radius, outer_radius, height, nrefine=None):
    """Creates a periodic hollow cylinder of with defined inner and outer
    radius along the z axis with given height.

    Parameters
    ----------
    inner_radius : float
        Inner radius
    outer_radius : float
        Outer radius
    height : float
        Length along z-axis

    Returns
    -------
    nutils.topology, nutils._Wrapper, nutils.function
        Domain, geometry and nurbs basis of nutils functions
    """

    domain, geom0 = mesh.rectilinear([4, 1, 1], periodic=[0])

    # Knot vector and knot multiplicities
    kv = [[0, 1, 2, 3, 4], [0, 1], [0, 1]]
    km = [[2, 2, 2, 2, 2], [2, 2], [2, 2]]

    bsplinebasis = domain.basis('spline',
                                degree=(2, 1, 1),
                                knotmultiplicities=km,
                                knotvalues=kv,
                                periodic=[0])

    cps = np.array([[inner_radius, 0, 0], [outer_radius, 0, 0],
                    [inner_radius, 0, height], [outer_radius, 0, height],
                    [inner_radius, -inner_radius, 0],
                    [outer_radius, -outer_radius, 0],
                    [inner_radius, -inner_radius, height],
                    [outer_radius, -outer_radius,
                     height], [0, -inner_radius, 0], [0, -outer_radius, 0],
                    [0, -inner_radius, height], [0, -outer_radius, height],
                    [-inner_radius, -inner_radius, 0],
                    [-outer_radius, -outer_radius, 0],
                    [-inner_radius, -inner_radius, height],
                    [-outer_radius, -outer_radius, height],
                    [-inner_radius, 0, 0], [-outer_radius, 0, 0],
                    [-inner_radius, 0, height], [-outer_radius, 0, height],
                    [-inner_radius, inner_radius, 0],
                    [-outer_radius, outer_radius, 0],
                    [-inner_radius, inner_radius, height],
                    [-outer_radius, outer_radius, height],
                    [0, inner_radius, 0], [0, outer_radius, 0],
                    [0, inner_radius, height], [0, outer_radius, height],
                    [inner_radius, inner_radius, 0],
                    [outer_radius, outer_radius, 0],
                    [inner_radius, inner_radius, height],
                    [outer_radius, outer_radius, height]])

    controlweights = np.tile(np.repeat([1., 1 / np.sqrt(2)], 4), 4)

    # Create nurbsbasis and geometry
    weightfunc = bsplinebasis.dot(controlweights)
    nurbsbasis = bsplinebasis * controlweights / weightfunc
    geom = (nurbsbasis[:, np.newaxis] * cps).sum(0)

    # Refine domain nrefine times
    if nrefine:
        domain = domain.refine(nrefine)
        bsplinebasis = domain.basis('spline', degree=2)
        controlweights = domain.project(weightfunc,
                                        onto=bsplinebasis,
                                        geometry=geom0,
                                        ischeme='gauss9')
        nurbsbasis = bsplinebasis * controlweights / weightfunc

    return domain, geom, nurbsbasis


def main(nrefine=1,
         poisson=0.3,
         diffusivity=0.01,
         thermal_expansion=0.01,
         timestep=0.1,
         endtime=0.2):
    """The function simulates the cooling of a cylinder with non-homogeneous
    temperature. The solution of the heat equation with Dirichlet boundary
    conditions serves as initial values for a thermoelastic problem.

    Parameters
    ----------
    nrefine : int, optional
        Number of refinements of the domain, by default 1
    poisson : float, optional
        poisson ration of the material, by default 0.3
    diffusivity : float, optional
        diffusivity of the material, by default 0.01
    thermal_expansion : float, optional
        thermal expansion of the material, by default 0.01
    timestep : float, optional
        timestep in the solution of the heat problem, by default 0.1
    endtime : float, optional
        endtime of the heat problem, by default 0.2
    """

    ns = Namespace()

    # Create a hollow cylinder as geometry
    domain, geom, nurbsbasis = get_cylinder(inner_radius=1.,
                                            outer_radius=1.5,
                                            height=2.0,
                                            nrefine=nrefine)
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS', 'dL'))
    integration_degree = 4

    # Heat equation
    ns.tbasis = nurbsbasis
    ns.temperature = function.dotarg('t', ns.tbasis)
    ns.k = diffusivity
    ns.boundaryT = 5.0

    # Weak form of the heat equation
    tres = domain.integral('k ∇_i(tbasis_n) ∇_i(temperature) dV' @ ns,
                           degree=integration_degree)
    tinertia = domain.integral('tbasis_n temperature dV' @ ns,
                               degree=integration_degree)

    # Enforce temperature on the boundary, which increases with height
    sqr = domain.boundary.integral('(temperature - boundaryT x_2)^2 dS' @ ns,
                                   degree=integration_degree)
    tcons = solver.optimize('t', sqr, droptol=1e-12)

    boundary_constrains = domain.boundary.project(fun='boundaryT x_2' @ ns,
                                                  onto=ns.tbasis,
                                                  geometry=geom,
                                                  ischeme='gauss9')
    lhs0 = domain.project(fun=0.,
                          onto=ns.tbasis,
                          geometry=geom,
                          constrain=boundary_constrains,
                          ischeme='gauss9')
    solution = []

    # Solve the heat problem
    with treelog.iter.plain(
            'timestep',
            solver.impliciteuler('t',
                                 tres,
                                 tinertia,
                                 lhs0=lhs0,
                                 timestep=timestep,
                                 constrain=tcons)) as steps:
        for itime, t_solution in enumerate(steps):

            solution.append({
                'timestep': itime * timestep,
                'solution': t_solution
            })

            if itime * timestep >= endtime:
                break

    # Elasticity problem
    ns.ubasis = nurbsbasis.vector(domain.ndims)
    ns.u = function.dotarg('u', ns.ubasis)

    ns.lmbda = 2 * poisson
    ns.mu = 1 - poisson
    ns.alpha = thermal_expansion
    # Final temperature
    ns.finalT = 0.0

    ns.δ = function.eye(domain.ndims)
    ns.X_i = 'x_i + u_i'
    ns.e_ij = '((∇_j(u_i) + ∇_i(u_j)) / 2) - alpha (temperature - finalT) δ_ij'
    ns.c_ijkl = 'lmbda δ_ij δ_kl + mu (δ_ik δ_jl + δ_il δ_jk)'
    ns.stress_ij = 'c_ijkl e_kl'

    # Constrain the inner radius on the bottom to restrict rigid body movement
    sqr = domain.boundary['bottom'].boundary['front'].integral(
        'u_i u_i dL' @ ns, degree=integration_degree)
    ucons = solver.optimize('u', sqr, droptol=1e-12)

    # Solve the momentum balance linearly
    ures = domain.integral('∇_j(ubasis_ni) stress_ij dV' @ ns,
                           degree=integration_degree)

    u_solution = solver.solve_linear('u',
                                     ures,
                                     constrain=ucons,
                                     arguments={'t': t_solution})

    # Sample the solution and export VTK
    bezier = domain.boundary.sample('bezier', 5)
    x, X, initialT, stress, normU = bezier.eval(
        [ns.x, ns.X, ns.temperature, ns.stress,
         function.norm2(ns.u)],
        u=u_solution,
        t=t_solution)
    export.vtk('deformed_cylinder', bezier.tri, X, initialT=initialT, u=normU)

    # Export matplotlib
    with export.mplfigure('displacement.png') as fig:
        ax = fig.add_subplot(111, projection='3d')

        meanU = np.array([np.mean(normU[t]) for t in bezier.tri])
        norm = Normalize(np.min(meanU), np.max(meanU))
        surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=bezier.tri)
        surf.set_fc(coolwarm(norm(meanU)))

        cbar = fig.colorbar(ScalarMappable(cmap=coolwarm, norm=norm))
        cbar.set_label('Displacement')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    return t_solution, u_solution


@testing.requires('matplotlib')
class test(testing.TestCase):
    def test_default(self):
        lhs_t, lhs_u = main(nrefine=1,
                            poisson=.3,
                            diffusivity=0.01,
                            thermal_expansion=0.1,
                            timestep=0.1,
                            endtime=0.1)

        with self.subTest('temperature'):
            self.assertAlmostEqual64(
                lhs_t, '''
            eNpjYPj/n4nh739Jy5eGL4wkLe2sV5nkmdpZn4NCJgYGhn//WRlwyQOlGRiBZuC
            SZwbKgVTgkv8Hlv2H136QC3HJQ0z/T0A/M079AJo7YXc=
    ''')
        with self.subTest('displacement'):
            self.assertAlmostEqual64(
                lhs_u, '''
            eNpF0a9OM0EcheFNEDhE76CmgaTdme3+ZtahUE1GgEJyAU1oqiqRBEEQXAAJpqqISVB
            FgNqdP9sZcBgcDhKaIErS5ONsPtJzAY94T5JgIbKXj3gWTuKDV2E/fjnNYy/LqJ/wn/
            QuGzJev/bnnPs3IeiCSI7pPf8Uq7zeO5KKdv291HQrd4o1ncqrYiIehSum4puS/37sw
            r9J4T+l8Nlz4wf41wH+wQK+tfAvLfxtB19b+MrC37Lwzw38Y/PnsxeefsSTeMYe/H7U
            7MvFnlpkWcKd/0mH7LDm9Zx3PPeC2uaCxuSq93yVt0y9p2hW7XpNg+pWrmlZnsqJGJW
            PYiqo/PNjF/5NCv8phc+eGz/Avw7wDxbwrYV/aeFvO/jawlcW/paFf27gH5uNz9OmD0
            MfzdBHLdDHefQ5rNGn49GnbdDHVejTMugzq9BnUKHPskSfUYk+VG768LT5l+FfzfCvW
            uDfX5/Dx78dj3/bBv+6Cv+2DP6dVfh3UOHfZYl/RyX+pXLzb+yG2PQJ6KMC+miOPtRH
            n7sMfV776PMm0Ick+nwK9DmS6HMv0WenQJ+rAn1csfkX/u+/Af/Cj73G5/CHDP6cwxc
            Ef0zwVzl8RfA1wV8T/ImAPxXw/wHBYmrs
    ''')

    def test_refined(self):
        lhs_t, lhs_u = main(nrefine=2,
                            poisson=.3,
                            diffusivity=0.01,
                            thermal_expansion=0.1,
                            timestep=0.1,
                            endtime=0.1)

        with self.subTest('temperature'):
            self.assertAlmostEqual64(
                lhs_t, '''
            eNr7/58BChgZDph/M265cPS8kckB82eWJmYzLtw6l2v2zPKplax5/dltZyLNn1r1Wv8
            w/294U1fXotf6AhJkZPgPNen/f0rM+fefkQFmziHzH8aNFw6dNzE5ZP7C0sxs+oWb5/
            LNXlg+t5I3rzm7+Uy0+XOrfuvf5ixGH3T1LfqtLyFBJrh7GBkoMYcBDijzFwMSoCycE
            YASf8HC+f9/ysKHkUrhg0g9lKVDBiqFD7XMoZa/
            GOG5lHrpcDCEDwDVFPye
    ''')
        with self.subTest('displacement'):
            self.assertAlmostEqual64(
                lhs_u, '''
            eNpF1n9oVfUfx/FpauZmGkUuR7pyRvMez+eez4+7MZgwZVGoq4UzpAyD/rBMmRhhihL
            GVkSSlEZQoGQhumgwJcWhA0dj95x7zv2czu0SNGtbLFfUNytnpmbf+4xy56/z7xNe5/
            E+ZWU8rydXi6bYGe8trIy77ZtJZ36TXZRsjjbammRqtMsZrb3geOIWZ2NtRtxIDcf14
            mLqdL5R2NTuqDHd7yyOxr0Ksc17S74i5ntPynrxnFgh/3YH4ow87lbnR2SLeCOarWd5
            Y2qHbvX2q8f1PC+RS/VQ+lGvSh9Kl8cH9K3etajPDMit+roZlfX6V3NOdqhLplveJ38
            1HfIxKzLH5MHwsNktPzMTpkueNM9kblObzfHMGlVmptbtVyt1qu6IalFl/3bNqKVrW4
            quFx26zjt0DTl0LSjS9VdC19QCXXMLdKULdDUmdF2y/3RZugYtXVNiulbHdPVYui5Hd
            HVHdP0Q0VWTp2tDnq4/I7oaQroWhHQtC+lqDenaGdK1NqRrT0hXW0jXHzm6PsnR9UKO
            rnW5/7qGkupUWXEkblhaF8+Iq9zt+T77rLshOms3uxPhIedoMmvpdNEVm9oK0RN/Hs8
            Vp+Oj+btFb9weTU9n7byo15uV3+Otk1uiB7wl8sNoi7hT9kZH42lyIJqd/0guil6Kiu
            qX3P/UMv1a7n1VqbtzX8kJVcg96o2qsdy3tk2350aiDrMq2KX7zO3Bcn3CzA/2qpOmM
            qiRPeae4EH7k/na3xs+b9YGfeaUmRmcNdWZw/5W055p86eYM5mZfrP+LvNNtvHfruFC
            jVNWHPhioVsXj8VXS10fxE1iQ/ROvFxMhE84G0pdbe41a2pb3Ev28/gh96I9mm92f7T
            tUZ24lp8XXU43R3u8t70j4QPe014x3CKavZ/Do3GDdz2cnT/vPRW+FN2hlpS6Xlbngv
            dVqxoPvpKOKi91VakFpa59qisYiU7onf4u/Ztu8pfr7/Uqf68a14/4NfKCbvEftIvNX
            aWuY/pdv89c0Q/7Z816M5zdarrMvuwUc8WsyDbr+zP33uy6WtzumOLewg53Zfxm0io6
            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
            4UZ3BjfYMbpzJ4MZ3Gdz4P8QqYts=
    ''')


if __name__ == '__main__':
    cli.run(main)
