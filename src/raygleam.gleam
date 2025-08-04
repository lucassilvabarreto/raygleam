import gleam/bool
import gleam/float
import gleam/io
import gleam/list
import gleam/result
import gleam/string

/// An opaque type representing a vector of floats in any dimension
pub opaque type Vector {
  Vector(inner: List(Float))
}

/// Creates a 2D vector from two floats
pub fn vector2d(x: Float, y: Float) -> Vector {
  Vector([x, y])
}

/// Creates a 3D vector from three floats
pub fn vector3d(x: Float, y: Float, z: Float) -> Vector {
  Vector([x, y, z])
}

pub fn vector4d(x: Float, y: Float, z: Float, w: Float) -> Vector {
  Vector([x, y, z, w])
}

/// Creates a unit basis vector with the specified dimension and index
/// The component at the given index will be 1.0, all others will be 0.0
///
/// # Examples
/// - `unit_basis(3, 0)` creates [1.0, 0.0, 0.0]
/// - `unit_basis(3, 1)` creates [0.0, 1.0, 0.0]
/// - `unit_basis(3, 2)` creates [0.0, 0.0, 1.0]
pub fn unit_basis(dimension: Int, index: Int) -> Vector {
  let components =
    list.range(0, dimension - 1)
    |> list.map(fn(i) {
      case i == index {
        True -> 1.0
        False -> 0.0
      }
    })

  Vector(components)
}

/// Converts a vector to a string representation for display
pub fn to_string(vec: Vector) -> String {
  let Vector(components) = vec
  let component_strings =
    components
    |> list.map(float.to_string)
    |> string.join(", ")

  "[" <> component_strings <> "]"
}

/// Checks if two vectors are equal by comparing their corresponding elements
pub fn equal_to(vec1: Vector, vec2: Vector) -> Bool {
  let Vector(components1) = vec1
  let Vector(components2) = vec2

  components1 == components2
}

/// Assigns components of vector y to the corresponding components of vector x
/// Returns a new vector with the components from y (essentially copying y)
pub fn becomes(x: Vector, y: Vector) -> Vector {
  let Vector(_) = x
  // We acknowledge x but don't use its components
  let Vector(y_components) = y

  Vector(y_components)
}

/// Adds two vectors by adding their corresponding components
/// Both vectors must have the same dimension
pub fn vector_addition(vec1: Vector, vec2: Vector) -> Vector {
  let Vector(components1) = vec1
  let Vector(components2) = vec2

  let added_components =
    list.zip(components1, components2)
    |> list.map(fn(pair) {
      let #(x, y) = pair
      x +. y
    })

  Vector(added_components)
}

pub fn vector_subtraction(vec1: Vector, vec2: Vector) -> Vector {
  let Vector(components1) = vec1
  let Vector(components2) = vec2

  let added_components =
    list.zip(components1, components2)
    |> list.map(fn(pair) {
      let #(x, y) = pair
      x -. y
    })

  Vector(added_components)
}

pub fn vector_scaling(vec: Vector, scalar: Float) -> Vector {
  let Vector(components) = vec

  let scaled_components =
    components
    |> list.map(fn(x) { x *. scalar })

  Vector(scaled_components)
}

pub fn axpy(a: Float, x: Vector, y: Vector) -> Vector {
  vector_scaling(x, a)
  |> vector_addition(y)
}

pub fn linear_combination(a: Float, x: Vector, b: Float, y: Vector) -> Vector {
  let scaled_x = vector_scaling(x, a)
  let scaled_y = vector_scaling(y, b)
  vector_addition(scaled_x, scaled_y)
}

pub fn dot_product(vec1: Vector, vec2: Vector) -> Float {
  let Vector(components1) = vec1
  let Vector(components2) = vec2

  list.zip(components1, components2)
  |> list.map(fn(pair) {
    let #(x, y) = pair
    x *. y
  })
  |> list.fold(0.0, fn(x, acc) { acc +. x })
}

pub fn length(vec: Vector) -> Float {
  Ok(dot_product(vec, vec))
  |> result.try(float.square_root)
  |> result.unwrap(0.0)
}

pub fn main() {
  let a = vector2d(4.0, -3.0)
  let b = vector2d(4.0, -3.0)
  io.println("Vector a: " <> to_string(a))
  io.println("Length of a: " <> float.to_string(length(a)))

  // Create some unit basis vectors
  let i = unit_basis(3, 0)
  // [1.0, 0.0, 0.0]
  let j = unit_basis(3, 1)
  // [0.0, 1.0, 0.0]
  let k = unit_basis(3, 2)
  // [0.0, 0.0, 1.0]

  io.println("Unit basis vector i: " <> to_string(i))
  io.println("Unit basis vector j: " <> to_string(j))
  io.println("Unit basis vector k: " <> to_string(k))

  // Test equality function
  io.println("i equals j: " <> bool.to_string(equal_to(i, j)))
  io.println("a equals b: " <> bool.to_string(equal_to(a, b)))

  // Test becomes function
  let c = vector2d(1.0, 2.0)
  let d = becomes(c, a)
  // d becomes the same as a
  io.println("Vector c: " <> to_string(c))
  io.println("Vector d after c becomes a: " <> to_string(d))

  // Test add function
  let x = vector2d(3.0, 4.0)
  let y = vector2d(1.0, 2.0)
  let z = vector_addition(x, y)
  io.println("Vector 1: " <> to_string(x))
  io.println("Vector 2: " <> to_string(y))
  io.println("Sum: " <> to_string(z))

  // Test vector scaling
  let scaled = vector_scaling(x, -0.5)
  io.println("Scaled vector: " <> to_string(scaled))

  // Test subtract function
  let x = vector2d(3.0, 4.0)
  let y = vector2d(1.0, 2.0)
  let z = vector_subtraction(x, y)
  io.println("Vector 1: " <> to_string(x))
  io.println("Vector 2: " <> to_string(y))
  io.println("Difference: " <> to_string(z))

  // test axpy function
  let a = 2.0
  let x = vector2d(1.0, 2.0)
  let y = vector2d(3.0, 4.0)
  let result = axpy(a, x, y)
  io.println("AXPY result: " <> to_string(result))

  // test linear combination function
  let a = 1.0
  let x = vector2d(1.0, 2.0)
  let b = 2.0
  let y = vector2d(3.0, 4.0)
  let linear_result = linear_combination(a, x, b, y)
  io.println("Linear combination result: " <> to_string(linear_result))

  // test dot product function
  let dot_result = dot_product(x, y)
  io.println(
    "Dot product of "
    <> to_string(x)
    <> " and "
    <> to_string(y)
    <> ": "
    <> float.to_string(dot_result),
  )

  let vector = vector4d(0.5, 0.5, 0.5, 0.5)
  io.println("Vector 4D: " <> to_string(vector))
  io.println("Length of 4D vector: " <> float.to_string(length(vector)))
}
