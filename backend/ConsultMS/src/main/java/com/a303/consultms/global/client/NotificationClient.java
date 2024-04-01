package com.a303.consultms.global.client;

import com.a303.consultms.global.api.response.BaseResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@FeignClient(name = "notificationms")
public interface NotificationClient {

	@PostMapping("/api/notifications/v0/consult")
	BaseResponse<Integer> consultNotification(
		@RequestParam("type") String type,
		@RequestParam("memberId") int memberId
	);

}