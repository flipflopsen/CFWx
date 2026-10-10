/**
 * @file include/util/colors.h
 * @brief colors interfaces and implementation.
 * @ingroup utilities
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for colors. */

#ifndef COLORS_H
#define COLORS_H

/**
 *
 * ANSI color definitions for pretty-print
 *
 */

/** @def NO_COLOR
 * @brief no color macro; see expansion and module contract.
 */
#define NO_COLOR "\x1b[0m"
/** @def RED
 * @brief red macro; see expansion and module contract.
 */
#define RED "\x1b[31m"
/** @def GREEN
 * @brief green macro; see expansion and module contract.
 */
#define GREEN "\x1b[32m"
/** @def BROWN
 * @brief brown macro; see expansion and module contract.
 */
#define BROWN "\x1b[33m"
/** @def BLUE
 * @brief blue macro; see expansion and module contract.
 */
#define BLUE "\x1b[34m"
/** @def MAGENTA
 * @brief magenta macro; see expansion and module contract.
 */
#define MAGENTA "\x1b[35m"
/** @def CYAN
 * @brief cyan macro; see expansion and module contract.
 */
#define CYAN "\x1b[36m"
/** @def GRAY
 * @brief gray macro; see expansion and module contract.
 */
#define GRAY "\x1b[37m"

#endif // COLORS_H
