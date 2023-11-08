from datetime import datetime

from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp

from action import Action


class CpuAction(Action):
    def perform(self, *args, **kwargs):
        start = datetime.now()

        def f(t, y, args):
            return -y

        term = ODETerm(f)
        solver = Dopri5()
        y0 = jnp.array([2., 3.])
        result = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)
        #print(f"CPU computation: {result} took: {(datetime.now() - start).total_seconds()}")
