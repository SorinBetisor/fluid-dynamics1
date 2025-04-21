using System;
using Unity.Mathematics;
using UnityEngine;

public class FluidSimulation : MonoBehaviour
{
    [Header("Simulation Parameters")]
    public int n = 64;

    public float diffusion = 0.0005f;
    public float viscosity = 0.0005f;
    public float dt = 0.1f;
    
    private FluidSolver _solver;
    
    private void Awake()
    {
        _solver = new FluidSolver(n, diffusion, viscosity, dt);
    }

    private void Update()
    {
        // Inject rotating force and density at center
        _solver.ResetSources();

        var cx = n / 2;
        var cy = n / 2;

        _solver.S[cx, cy] = 100f;
        _solver.Vx0[cx, cy] = 20f * Mathf.Sin(Time.time);
        _solver.Vy0[cx, cy] = 20f * Mathf.Cos(Time.time);

        _solver.Step();
    }
}

public class FluidSolver
{
    /// <summary>
    /// Number of particles
    /// </summary>
    private readonly int _n;

    /// <summary>
    /// Grid size
    /// </summary>
    private readonly int _gridSize;

    /// <summary>
    /// Timestep
    /// </summary>
    private readonly float _dt;

    /// <summary>
    /// Density-Diffusion coefficient
    /// </summary>
    private readonly float _diffusion;

    /// <summary>
    /// Kinematic-Viscosity coefficient
    /// </summary>
    private readonly float _viscosity;

    /// <summary>
    /// Amount of iterations to solve a PDE
    /// </summary>
    private readonly int _iterations;

    public readonly float[,] S, Density;
    public readonly float[,] Vx0, Vy0;

    private readonly float[,] _vx, _vy;

    public FluidSolver(int n, float dt, float diffusion, float viscosity, int iterations = 20)
    {
        _n = n;
        _dt = dt;
        _diffusion = diffusion;
        _viscosity = viscosity;
        _iterations = iterations;

        _gridSize = _n + 2;

        S = new float[_gridSize, _gridSize];
        Density = new float[_gridSize, _gridSize];

        _vx = new float[_gridSize, _gridSize];
        _vy = new float[_gridSize, _gridSize];

        Vx0 = new float[_gridSize, _gridSize];
        Vy0 = new float[_gridSize, _gridSize];
    }

    public void Step()
    {
        AddSource(_vx, Vx0);
        AddSource(_vy, Vy0);
        Project(_vx, _vy, Vx0, Vy0);

        Diffuse(1, Vx0, _vx, _viscosity);
        Diffuse(2, Vy0, _vy, _viscosity);
        Project(Vx0, Vy0, _vx, _vy);

        Advect(1, _vx, Vx0, Vy0, Vy0);
        Advect(2, _vy, Vy0, Vx0, Vy0);
        Project(_vx, _vy, Vx0, Vy0);

        AddSource(Density, S);
        Diffuse(0, Density, S, _diffusion);
        Advect(0, Density, S, _vx, _vy);
    }

    public void ResetSources()
    {
        Array.Clear(S, 0, S.Length);
        Array.Clear(Vx0, 0, Vx0.Length);
        Array.Clear(Vy0, 0, Vy0.Length);
    }

    private void AddSource(float[,] x, float[,] s)
    {
        for (var i = 0; i < _gridSize; i++)
        for (var j = 0; j < _gridSize; j++)
            x[i, j] += _dt * s[i, j];
    }

    private void Diffuse(int b, float[,] x, float[,] x0, float diff)
    {
        var a = _dt * diff * _n * _n;
        LinearSolve(b, x, x0, a, 1 + 4 * a);
    }

    private void Advect(int b, float[,] d, float[,] d0, float[,] u, float[,] v)
    {
        var dt0 = _dt * _n;

        for (var i = 1; i <= _n; i++)
        {
            for (var j = 1; j <= _n; j++)
            {
                var x = i - dt0 * u[i, j];
                var y = j - dt0 * v[i, j];

                x = math.clamp(x, 0.5f, _n + 0.5f);
                y = math.clamp(y, 0.5f, _n + 0.5f);
                
                var i0 = (int)math.floor(x);
                var i1 = i0 + 1;

                var j0 = (int)math.floor(y);
                var j1 = j0 + 1;

                var s1 = x - i0;
                var s0 = 1 - s1;

                var t1 = y - j0;
                var t0 = 1 - t1;

                d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1])
                          + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]);
            }
        }

        SetBoundary(b, d);
    }

    private void Project(float[,] u, float[,] v, float[,] p, float[,] div)
    {
        var h = 1f / _n;

        for (var i = 1; i <= _n; i++)
        {
            for (var j = 1; j <= _n; j++)
            {
                div[i, j] = -0.5f * h * (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1]);
                p[i, j] = 0;
            }
        }

        SetBoundary(0, div);
        SetBoundary(0, p);

        LinearSolve(0, p, div, 1, 4);

        for (var i = 1; i <= _n; i++)
        {
            for (var j = 1; j <= _n; j++)
            {
                u[i, j] -= 0.5f * (p[i + 1, j] - p[i - 1, j]) / h;
                v[i, j] -= 0.5f * (p[i, j + 1] - p[i, j - 1]) / h;
            }

            SetBoundary(1, u);
            SetBoundary(2, v);
        }
    }

    private void LinearSolve(int b, float[,] x, float[,] x0, float a, float c)
    {
        for (var k = 0; k < _iterations; k++)
        {
            for (var i = 1; i <= _n; i++)
            {
                for (var j = 1; j <= _n; j++)
                {
                    x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / c;
                }
            }

            SetBoundary(b, x);
        }
    }

    private void SetBoundary(int b, float[,] x)
    {
        for (var i = 0; i <= _n; i++)
        {
            x[0, i] = b == 1 ? -x[1, i] : x[0, i];
            x[_n + 1, i] = b == 1 ? -x[_n, i] : x[_n, i];
            x[i, 0] = b == 2 ? -x[i, 1] : x[i, 1];
            x[i, _n + 1] = b == 2 ? -x[i, _n] : x[i, _n];
        }

        x[0, 0] = 0.5f * (x[1, 0] + x[0, 1]);
        x[0, _n + 1] = 0.5f * (x[1, _n + 1] + x[0, _n]);
        x[_n + 1, 0] = 0.5f * (x[_n, 0] + x[_n + 1, 1]);
        x[_n + 1, _n + 1] = 0.5f * (x[_n, _n + 1] + x[_n + 1, _n]);
    }
}