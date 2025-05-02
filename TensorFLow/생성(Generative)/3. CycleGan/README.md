<!--
fake_y = X(말)를 Y(얼룩말)로 즉, G(X) → Y'
cycled_x = Y(얼룩말)을 X(말)로 즉, F(G(X)) → X' (cycle consistency)

fake_x = Y(얼룩말)를 X(말)로 즉, F(Y) → X'
cycled_y = X(말)를 Y(얼룩말)로 즉, G(F(Y)) → Y' (cycle consistency)

동일한 이미지를 변환기에 넣었을 때 이미지가 바뀌지 않고 유지되도록 하는 identity loss 계산용
- same_x = loss 계산용
- smae_y = loss 계산용


























참고고
# Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
-->