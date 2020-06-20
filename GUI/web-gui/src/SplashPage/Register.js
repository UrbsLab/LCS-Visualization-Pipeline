import React from "react";
import {
  Dialog,
  TextField,
  Button,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@material-ui/core";

class RegisterDialog extends React.Component {
  render() {
    return (
      <Dialog
        open={this.props.shouldDisplay}
        onClose={this.props.close}
        aria-labelledby="form-dialog-title"
      >
        <DialogTitle id="form-dialog-title">Register for Apollo</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Get started with your Apollo Dashboard here
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            id="firstname"
            label="first name"
            type="text"
            fullWidth
          />
          <TextField
            autoFocus
            margin="dense"
            id="lastname"
            label="last name"
            type="text"
            fullWidth
          />
          <TextField
            autoFocus
            margin="dense"
            id="username"
            label="email address"
            type="email"
            fullWidth
          />
          <TextField
            autoFocus
            margin="dense"
            id="password"
            label="password"
            type="password"
            fullWidth
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={this.props.register} color="primary">
            Register
          </Button>
        </DialogActions>
      </Dialog>
    );
  }
}

export default RegisterDialog;
