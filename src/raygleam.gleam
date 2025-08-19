import gleam/bool
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string

/// An opaque type representing a vector of floats in any dimension
pub opaque type Vector {
  Vector(inner: List(Float))
}

/// An opaque type representing a matrix as a collection of column vectors
pub opaque type Matrix {
  Matrix(columns: List(Vector))
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

/// Creates a matrix from a list of column vectors
pub fn matrix_from_columns(columns: List(Vector)) -> Matrix {
  Matrix(columns)
}

/// Creates a matrix from a list of lists, where each inner list represents a column
pub fn matrix_from_lists(column_lists: List(List(Float))) -> Matrix {
  let columns =
    column_lists
    |> list.map(fn(column) { Vector(column) })

  Matrix(columns)
}

/// Gets the column vectors from a matrix
pub fn get_columns(matrix: Matrix) -> List(Vector) {
  let Matrix(columns) = matrix
  columns
}

/// Gets a specific column vector by index (0-based)
pub fn get_column(matrix: Matrix, index: Int) -> Result(Vector, Nil) {
  let Matrix(columns) = matrix

  columns
  |> list.drop(index)
  |> list.first
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

/// Converts a matrix to a string representation for display
pub fn matrix_to_string(matrix: Matrix) -> String {
  let Matrix(columns) = matrix
  let column_strings =
    columns
    |> list.map(to_string)
    |> string.join(" ")

  "[" <> column_strings <> "]"
}

/// Gets the number of columns in a matrix
pub fn matrix_columns(matrix: Matrix) -> Int {
  let Matrix(columns) = matrix
  list.length(columns)
}

/// Gets the number of rows in a matrix (based on the first column)
/// Returns 0 if the matrix has no columns
pub fn matrix_rows(matrix: Matrix) -> Int {
  let Matrix(columns) = matrix
  case columns {
    [] -> 0
    [Vector(first_column), ..] -> list.length(first_column)
  }
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

/// Multiplies a matrix by a vector (matrix * vector)
/// The number of columns in the matrix must match the dimension of the vector
pub fn matrix_vector_multiply(
  matrix: Matrix,
  vec: Vector,
) -> Result(Vector, String) {
  let Matrix(columns) = matrix
  let Vector(vec_components) = vec

  // Check if dimensions match
  case list.length(columns) == list.length(vec_components) {
    False -> Error("Matrix columns must match vector dimension")
    True -> {
      // For each column, scale it by the corresponding vector component
      let scaled_columns =
        list.zip(columns, vec_components)
        |> list.map(fn(pair) {
          let #(column, scalar) = pair
          vector_scaling(column, scalar)
        })

      // Sum all the scaled columns
      case scaled_columns {
        [] -> Error("Empty matrix")
        [first, ..rest] -> {
          let result = list.fold(rest, first, vector_addition)
          Ok(result)
        }
      }
    }
  }
}

/// Creates an identity matrix of the specified size
pub fn identity_matrix(size: Int) -> Matrix {
  let columns =
    list.range(0, size - 1)
    |> list.map(fn(i) { unit_basis(size, i) })

  Matrix(columns)
}

/// Gets a specific row from a matrix by index (0-based)
/// Returns a vector containing the elements from the specified row across all columns
pub fn get_row(matrix: Matrix, row_index: Int) -> Result(Vector, String) {
  let Matrix(columns) = matrix

  case columns {
    [] -> Error("Empty matrix")
    _ -> {
      let row_components =
        columns
        |> list.map(fn(column) {
          let Vector(components) = column
          components
          |> list.drop(row_index)
          |> list.first
          |> result.unwrap(0.0)
        })

      Ok(Vector(row_components))
    }
  }
}

/// Transposes a matrix (swaps rows and columns)
pub fn transpose(matrix: Matrix) -> Matrix {
  let Matrix(columns) = matrix

  case columns {
    [] -> Matrix([])
    _ -> {
      let num_rows = matrix_rows(matrix)
      let transposed_columns =
        list.range(0, num_rows - 1)
        |> list.map(fn(row_idx) {
          case get_row(matrix, row_idx) {
            Ok(row_vector) -> row_vector
            Error(_) -> Vector([])
            // This shouldn't happen with valid indices
          }
        })

      Matrix(transposed_columns)
    }
  }
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

  // Test Matrix functionality
  io.println("\n--- Matrix Examples ---")

  // Create a 2x2 matrix using column vectors
  let col1 = vector2d(1.0, 2.0)
  let col2 = vector2d(3.0, 4.0)
  let matrix = matrix_from_columns([col1, col2])

  io.println("Matrix: " <> matrix_to_string(matrix))
  io.println(
    "Matrix dimensions: "
    <> string.join(
      [
        int.to_string(matrix_rows(matrix)),
        "x",
        int.to_string(matrix_columns(matrix)),
      ],
      "",
    ),
  )

  // Create matrix from lists
  let matrix2 = matrix_from_lists([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
  io.println("Matrix2: " <> matrix_to_string(matrix2))

  // Test matrix-vector multiplication
  let test_vec = vector2d(2.0, 3.0)
  case matrix_vector_multiply(matrix, test_vec) {
    Ok(result_vec) -> io.println("Matrix * Vector = " <> to_string(result_vec))
    Error(msg) -> io.println("Error: " <> msg)
  }

  // Create and display identity matrix
  let identity = identity_matrix(3)
  io.println("3x3 Identity Matrix: " <> matrix_to_string(identity))

  // Test getting a row from matrix
  case get_row(matrix, 0) {
    Ok(row) -> io.println("First row of matrix: " <> to_string(row))
    Error(msg) -> io.println("Error getting row: " <> msg)
  }

  // Test matrix transpose
  let transposed = transpose(matrix2)
  io.println("Original matrix2: " <> matrix_to_string(matrix2))
  io.println("Transposed matrix2: " <> matrix_to_string(transposed))
}
