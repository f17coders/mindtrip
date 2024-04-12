package com.a303.notificationms.domain.notification.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;

@Builder
public record CountNotificationMessageRes(
	String type,
	Long count,
	LocalDateTime localDateTime
) {

}
