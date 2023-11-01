/**
 *  @file   OrthogonalPolynomialSlidingWindow.hpp
 *  @brief  OrthogonalPolynomialSlidingWindow class header
 *  @date   26.03.2013
 *  @author Dr. Thiemo Gruber, Peter Rudolph
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES\n
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF\n
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR\n
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES\n
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN\n
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF\n
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.\n
 *
 *  @copyright Intelligent Embedded Systems, University of Kassel, 2013. All rights reserved.
 *  @license This library is released under <a href="http://www.gnu.org/licenses/lgpl-3.0.de.html">LGPLv3</a>.
 */

#ifndef __ORTHOGONALPOLYNOMIALSLIDINGWINDOW_HPP__
#define __ORTHOGONALPOLYNOMIALSLIDINGWINDOW_HPP__

#include <vector>

/**
 *  @brief This class implements an updatable sliding window least squares polynomial
 *         approximation using discrete Legendre polynomials.
 *
 *  @see   "Erich Fuchs, 'Schnelle Quadratmittelapproximation in gleitenden
 *         Zeitfenstern mit diskreten orthogonalen Polynomen', Ph.D. thesis,
 *         Universitaet Passau, Fakultaet fuer Mathematik und Informatik, 1999"
 */
class OrthogonalPolynomialSlidingWindow {

public:
  /**
   * @brief Constructor
   *
   * @param degree Degree of polynomial approximation
   * @param windowSize Size of sliding window
   */
	OrthogonalPolynomialSlidingWindow(int degree, int windowSize);

	/**
	 * @brief Initialize sliding window.
	 */
	void init();

	/**
	 * @brief Restart sliding window.
	 */
	void restart();

	/**
	 * @brief Update sliding window approximation.
	 * @param y Value to update with.
	 */
	void update(double y);

	/**
	 * @brief Evaluate polynomial at x.
	 * @param x Value to evaluate polynomial at.
	 * @return Evaluation
	 */
	double eval(double x);

	/**
	 * @brief Get orthogonal coefficients.
	 * @return Vector of orthogonal coefficients
	 */
	std::vector<double> getOrthogonalCoefficients();

  /**
   * @brief Get approximation error.
   * @return Approximation error
   */
	double getApproximationError();
  /**
   * @brief Get extremum position.
   * @return Extremum position
   */
	int getExtremumPosition();

private:
	int degree; /**< degree */
	int windowSize; /**< window size */
	std::vector<double> qnorm; /**< vector qnorm */
	std::vector<double> delta; /**< vector delta */
	std::vector<double> alpha; /**< vector alpha */
	std::vector<std::vector<double> > updateCoefs; /**< vector update coefficients */
	std::vector<double> values; /**< vector values */
	int valueIdx; /**< value idx  */
	double sqsum; /**< square sum */
};

#endif /* #ifndef __ORTHOGONALPOLYNOMIALSLIDINGWINDOW_H__*/

