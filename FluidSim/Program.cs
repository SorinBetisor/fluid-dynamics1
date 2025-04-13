using Flecs.NET.Core;

const int maxStepCount = 10;
const int timeStep = 1;

const int width = 10;
const int height = 10;

const float diffusionFactor = 1;

var grid = new Entity[width, height];

using var world = World.Create();
{
    var cellsQuery = world.Query<Density, Position, Velocity>();
    
    world.System("Diffusion")
        .Iter(_ =>
        {
            cellsQuery.Each((Iter iter, int i, ref Density d, ref Position p, ref Velocity v) =>
            {
                // d_X = (d0_X + diffusionFactor * deltaTime * (d_01 + d_02+ d_03 + d_04)) / (1 + 4 * diffusionFactor * deltaTime)
                d.Value = (d.Value + diffusionFactor * timeStep * ()) / (1 + 4 * diffusionFactor * timeStep);
            });
        });
    
    world.System<Density, Position, Velocity>("Display")
        .Each((Iter iter, int i, ref Density density, ref Position pos, ref Velocity vel) =>
        {
            Console.WriteLine($"Index: {i}, Density: {density}, Position: {pos}, Velocity: {vel}");
        });
}

{
    for (var i = 0; i < width; i++)
    {
        for (var j = 0; j < height; j++)
        {
            var entity = world.Entity()
                .Set(new Density(0))
                .Set(new Position(i, j))
                .Set(new Velocity(0, 0));
                
            grid[i, j] = entity;
        }
    }
}

{
    for (var i = 0; i < maxStepCount; i++)
    {
        Console.WriteLine($"----------STEP: {i * timeStep}----------");
        world.Progress();
    }
}

public record struct Density(float Value);
public record struct Position(int X, int Y);
public record struct Velocity(int X, int Y);