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

/// Computes the Euclidean length (norm) of the vector in any dimension
pub fn length(vec: Vector) -> Float {
  let Vector(components) = vec

  components
  |> list.map(fn(x) { x *. x })
  |> list.fold(Ok(0.0), fn(acc, x) { result.map(acc, fn(sum) { sum +. x }) })
  |> result.try(float.square_root)
  |> result.unwrap(0.0)
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
}
