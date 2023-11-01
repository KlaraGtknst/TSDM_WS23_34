/**
 *  @file   GrowingWindow.hpp
 *  @brief  GrowingWindow class header
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

#ifndef __GROWINGWINDOW_HPP__
#define __GROWINGWINDOW_HPP__

#include <vector>

/**
 *  @brief  This class provides algorithms for up- and downdating least squares
 *          polynomial fits to discrete data with orthogonal polynomials using
 *          (hyper-)rotations and for evaluating the resulting polynomials.
 *
 *  @see    Fuchs, E.; Gruber, T.; Nitschke, J.; Sick, B.,
 *          "Online Segmentation of Time Series Based on Polynomial Least-Squares Approximations,"
 *          Pattern Analysis and Machine Intelligence, IEEE Transactions on ,
 *          vol.32, no.12, pp.2232,2245, Dec. 2010
 *
 */
class GrowingWindow
{
public:
  /**
   * @brief Constructor
   *
   * @param degree Degree of polynom approximation
   */
  GrowingWindow(unsigned int degree);
  /**
   * @brief Destructor
   *
   */
  ~GrowingWindow();
  /**
   * @brief Restart growing window approximation.
   *
   */
  void reset();
  /**
   * @brief Update growing window approximation at y.
   *
   * @param y Update value
   */
  void update(double y);
  /**
   * @brief Evaluate polynomial at x.
   *
   * @param x Value polynomial is evaluated at.
   * @return Evaluation at x.
   */
  double eval(double x);
  /**
   * @brief Get number of samples.
   *
   * @return Number of samples.
   */
  int getNumSamples();
  /**
   * @brief Get orthogonal coefficients
   *
   * @return Vector containing orthogonal coefficients
   */
  std::vector<double> getOrthogonalCoefficients();
  /**
   * @brief Get approximation error
   *
   * @return Approximation error
   */
  double getApproximationError();
  /**
   * @brief Get extremum position
   *
   * @return Extremum position
   */
  int getExtremumPosition();

private:
  std::vector<double> alpha_; /**< alpha vector */
  std::vector<double> beta_;  /**< beta vector */
  std::vector<double> d_;     /**< d vector */
  int degree_;                /**< degree */
  int n_;                     /**< n */
  int nb_pts_;                /**< number of points */
  double sigma_;              /**< sigma */
  double ysqsum_;             /**< ysqsum */
};

#endif /* GW_H_ */
