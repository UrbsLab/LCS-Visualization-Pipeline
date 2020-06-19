import React from "react";
import { Typography } from "@material-ui/core";

class ExperimentContent extends React.Component {
  render() {
    return (
      <Typography variant="h5" align="center" style={{ margin: 20 }}>
        {this.props.name}
      </Typography>
    );
  }
}

export default ExperimentContent;
