import React from "react";
import "./Button.css";
import { Link } from "react-router-dom";

const STYLES = ["btn--primary", "btn--outline", "btn--red-primary", "btn--green-primary","btn--red-outline", "btn--green-outline"];

const SIZES = ["btn--medium", "btn--large"];

export const Button = ({
  children,
  type,
  id,
  onClick,
  buttonStyle,
  buttonSize,
  gridClass,
  hidebtn,
}) => {
  const checkButtonStyle = STYLES.includes(buttonStyle)
    ? buttonStyle
    : STYLES[0];

  const checkButtonSize = SIZES.includes(buttonSize) ? buttonSize : SIZES[0];

  return (
    <Link to="#" className={`btn-mobile ${gridClass}`}>
      <button
        className={`btn ${checkButtonStyle} ${checkButtonSize} `}
        id={id}
        onClick={onClick}
        type={type}
        style = {hidebtn}
      >
        {children}
      </button>
    </Link>
  );
};
