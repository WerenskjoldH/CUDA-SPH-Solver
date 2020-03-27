#ifndef VECTOR_2F_H
#define VECTOR_2F_H

#include <iostream>
#define PI 3.14159265f

typedef float real;

// This is being borrowed from my Ronin Math Library
class vector2f
{
	private:

	public:
		real x;
		real y;
		// For 4-word speed increase, most computers now-a-days use 4-word memory busses and, as such, just 3-words requires an offset
		real padding;

		// Constructors
		vector2f(real iX = 0, real iY = 0) : x(iX), y(iY) {};

		// Deconstructor
		~vector2f() {}

		// Copy Constructor
		vector2f(const vector2f& v) : x(v.x), y(v.y) {};

		// Operator Overloads
		vector2f& operator=(const vector2f& v)
		{
			x = v.x;
			y = v.y;

			return *this;
		}

		// Add components to eachother and set equal
		void operator+=(const vector2f& v)
		{
			x += v.x;
			y += v.y;
		}

		// Subtract components from eachother and set equal
		void operator-=(const vector2f& v)
		{
			x -= v.x;
			y -= v.y;
		}

		// Multiply vector by scalar and set equal
		void operator*=(const real& s)
		{
			x *= s;
			y *= s;
		}

		// Divide vector by scalar and set equal
		void operator/=(const real& s)
		{
			real mag = 1.f / s;
			x = x * mag;
			y = y * mag;
		}

		// Calculate scalar(dot) product and return result
		real operator*(const vector2f& v) const
		{
			return x * v.x + y * v.y;
		}

		// Divide by scalar and return result
		vector2f operator/(const real s) const
		{
			real mag = 1.f / s;
			return vector2f(x * mag, y * mag);
		}

		// Returns scalar(dot) product
		vector2f operator*(const real s) const
		{
			return vector2f(s * x, s * y);
		}

		// Subtract component-wise and return result
		vector2f operator-(const vector2f& v) const
		{
			return vector2f(x - v.x, y - v.y);
		}

		// Add component-wise and return result
		vector2f operator+(const vector2f& v) const
		{
			return vector2f(x + v.x, y + v.y);
		}

		// Negate components and return result
		vector2f operator-() const
		{
			return vector2f(-x, -y);
		}

		// Access components array-wise for modification
		real& operator[](int i)
		{
			switch (i)
			{
			case 1:
				return y;
				break;
			case 0:
				return x;
				break;
			default:
				std::cout << "ERROR::VECTOR::OUT OF BOUNDS REQUEST" << std::endl;
			}
		}

		// Access components array-wise for reading
		real operator[](int i) const
		{
			switch (i)
			{
			case 1:
				return y;
				break;
			case 0:
				return x;
				break;
			default:
				std::cout << "ERROR::VECTOR::OUT OF BOUNDS REQUEST" << std::endl;
			}
		}

		// Calculate the unit vector and return result
		vector2f unit() const
		{
			return vector2f(*this / this->magnitude());
		}

		// Invert components 
		void invert()
		{
			x *= -1;
			y *= -1;
		}

		// Normalize the vector and set equal
		void normalize()
		{
			(*this) = (*this).unit();
		}

		// Return magnitude of vector
		real magnitude() const
		{
			return std::sqrtf(x * x + y * y);
		}

		// Calculate squared magnitude and return result
		real squaredMagnitude()
		{
			return x * x + y * y;
		}

		// Borrowing this straight from Ian Millington
		// Add given vector scaled by float and set equal
		void addScaledVector(const vector2f& v, real t)
		{
			x += v.x * t;
			y += v.y * t;
		}

		real angleBetween(const vector2f& v) const
		{
			vector2f aU = this->unit();
			vector2f bU = v.unit();
			return acosf(aU * bU);
		}
};

#endif
